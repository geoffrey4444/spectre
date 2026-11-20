// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/ApparentHorizonFinder/ComputeCoords.hpp"

#include <algorithm>
#include <cstddef>
#include <deque>

#include "DataStructures/Tensor/IndexType.hpp"
#include "Domain/BlockLogicalCoordinates.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "NumericalAlgorithms/Strahlkorper/Strahlkorper.hpp"
#include "NumericalAlgorithms/Strahlkorper/StrahlkorperFunctions.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/FastFlow.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Storage.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/TimeExtrapolation.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace ah {
template <typename Fr>
bool set_current_iteration_coords(
    const gsl::not_null<ah::Storage::Iteration<Fr>*> current_iteration,
    const gsl::not_null<std::vector<size_t>*> block_order,
    const LinkedMessageId<double>& time, const FastFlow& fast_flow,
    const ylm::Strahlkorper<Fr>& initial_guess,
    const ylm::Strahlkorper<Fr>& previous_iteration_surface,
    const std::deque<ah::Storage::PreviousSurface<Fr>>& previous_surfaces,
    const size_t max_compute_coords_retries, const Domain<3>& domain,
    const domain::FunctionsOfTimeMap& functions_of_time,
    const std::optional<size_t>& current_resolution_l,
    const bool rerunning_with_higher_resolution,
    const size_t max_extrapolation_points) {
  // If we are rerunning the horizon find with higher resolution, use
  // the previous horizon as the initial guess.
  if (fast_flow.current_iteration() == 0) {
    if (rerunning_with_higher_resolution) {
      if (not current_resolution_l.has_value()) {
        ERROR(
            "Current resolution L is not set, even though this iteration is "
            "marked as a rerun with higher resolution. This is an internal "
            "error. Please file an issue.");
      }
      if (previous_iteration_surface.l_max() >= *current_resolution_l) {
        ERROR("Previous iteration surface has resolution "
              << previous_iteration_surface.l_max()
              << " but current resolution " << *current_resolution_l
              << " is not higher, even though this iteration is marked as a "
                 "rerun with higher resolution. This is an internal error. "
                 "Please file an issue.");
      }
      // Just set the initial guess by prolonging or restricting the previous
      // iteration surface, i.e., the horizon already found at a lower
      // resolution
      current_iteration->strahlkorper =
          ylm::Strahlkorper<Fr>(*current_resolution_l, *current_resolution_l,
                                previous_iteration_surface);
    } else {
      const auto& seed = previous_surfaces.empty()
                             ? initial_guess
                             : previous_surfaces.front().surface;
      auto& surface = current_iteration->strahlkorper;
      if (current_resolution_l.has_value() and
          *current_resolution_l != seed.l_max()) {
        surface = ylm::Strahlkorper<Fr>(*current_resolution_l,
                                        *current_resolution_l, seed);
      } else {
        surface = seed;
      }
      const size_t number_of_points =
          std::min(max_extrapolation_points, previous_surfaces.size());
      if (number_of_points >= 2) {
        surface.coefficients() = extrapolate_surface_coefficients(
            time.id, number_of_points, previous_surfaces,
            surface.ylm_spherepack());
      }
    }
  }  // if fastflow.current_iteration() == 0

  const auto set_coords = [&]() {
    const auto& strahlkorper = current_iteration->strahlkorper;

    const size_t l_mesh = fast_flow.current_l_mesh(strahlkorper);

    const auto prolonged_strahlkorper =
        ylm::Strahlkorper<Fr>(l_mesh, l_mesh, strahlkorper);

    // Frames are handled within block_logical_coordinates
    current_iteration->block_coord_holders = ::block_logical_coordinates(
        domain, ylm::cartesian_coords(prolonged_strahlkorper), time.id,
        functions_of_time, block_order);
  };

  // Set the coordinates
  set_coords();

  // Check if any points were outside the domain
  current_iteration->compute_coords_retries = 0;
  while (alg::any_of(
      current_iteration->block_coord_holders.value(),
      [](const auto& coord_holder) { return not coord_holder.has_value(); })) {
    // If so, try recomputation up to max_compute_coords_retries times
    ++current_iteration->compute_coords_retries;

    if (current_iteration->compute_coords_retries <=
        max_compute_coords_retries) {
      if (fast_flow.current_iteration() == 0) {
        // If this is the zeroth iteration and we couldn't compute the coords,
        // then just try increasing the size of the horizon by 50%
        current_iteration->strahlkorper.coefficients()[0] *= 1.5;
      } else {
        // Otherwise move the new trial surface halfway between the current
        // surface and the previous surface
        current_iteration->strahlkorper.coefficients() +=
            0.5 * (previous_iteration_surface.coefficients() -
                   current_iteration->strahlkorper.coefficients());
      }

      // Set the coords using the new guess
      set_coords();
    } else {
      // We didn't actually try computation here so we added one extra to the
      // counter which we need to remove
      --current_iteration->compute_coords_retries;
      return false;
    }
  }

  return true;
}

#define FRAME(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                            \
  template bool set_current_iteration_coords(                           \
      const gsl::not_null<ah::Storage::Iteration<FRAME(data)>*>         \
          current_iteration,                                            \
      const gsl::not_null<std::vector<size_t>*> block_order,            \
      const LinkedMessageId<double>& time, const FastFlow& fast_flow,   \
      const ylm::Strahlkorper<FRAME(data)>& initial_guess,              \
      const ylm::Strahlkorper<FRAME(data)>& previous_iteration_surface, \
      const std::deque<ah::Storage::PreviousSurface<FRAME(data)>>&      \
          previous_surfaces,                                            \
      const size_t max_compute_coords_retries, const Domain<3>& domain, \
      const domain::FunctionsOfTimeMap& functions_of_time,              \
      const std::optional<size_t>& current_resolution_l,                \
      const bool rerunning_with_higher_resolution,                      \
      const size_t max_extrapolation_points);

GENERATE_INSTANTIATIONS(INSTANTIATE,
                        (Frame::Inertial, Frame::Distorted, Frame::Grid))

#undef INSTANTIATE
#undef FRAME
}  // namespace ah
