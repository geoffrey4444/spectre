// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/ApparentHorizonFinder/OptionTags.hpp"

#include <cmath>
#include <cstddef>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#include "DataStructures/Tensor/IndexType.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "Options/Context.hpp"
#include "Options/ParseError.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"

namespace ah {
RescaledSurfaceCharSpeedOptions::RescaledSurfaceCharSpeedOptions(
    std::string excision_sphere_in, const size_t number_of_surfaces_in,
    const double relative_excision_margin_in, const Options::Context& context)
    : excision_sphere(std::move(excision_sphere_in)),
      number_of_surfaces(number_of_surfaces_in),
      relative_excision_margin(relative_excision_margin_in) {
  if (excision_sphere.empty()) {
    PARSE_ERROR(context, "ExcisionSphere must not be empty.");
  }
  if (number_of_surfaces < 2) {
    PARSE_ERROR(context, "NumberOfSurfaces must be at least 2, but is "
                             << number_of_surfaces << ".");
  }
  if (not std::isfinite(relative_excision_margin) or
      relative_excision_margin <= 0.0) {
    PARSE_ERROR(context,
                "RelativeExcisionMargin must be finite and positive, but is "
                    << relative_excision_margin << ".");
  }
}

void RescaledSurfaceCharSpeedOptions::pup(PUP::er& p) {
  p | excision_sphere;
  p | number_of_surfaces;
  p | relative_excision_margin;
}

bool operator==(const RescaledSurfaceCharSpeedOptions& lhs,
                const RescaledSurfaceCharSpeedOptions& rhs) {
  return lhs.excision_sphere == rhs.excision_sphere and
         lhs.number_of_surfaces == rhs.number_of_surfaces and
         lhs.relative_excision_margin == rhs.relative_excision_margin;
}

bool operator!=(const RescaledSurfaceCharSpeedOptions& lhs,
                const RescaledSurfaceCharSpeedOptions& rhs) {
  return not(lhs == rhs);
}

template <typename Fr>
HorizonOptions<Fr>::HorizonOptions(
    std::vector<std::unique_ptr<ah::Criterion>> criteria_in,
    ylm::Strahlkorper<Fr> initial_guess_in, ::FastFlow fast_flow_in,
    ::Verbosity verbosity_in, const size_t max_compute_coords_retries_in,
    std::optional<std::vector<std::string>> blocks_for_horizon_find_in,
    std::optional<RescaledSurfaceCharSpeedOptions>
        rescaled_surface_char_speeds_in,
    const Options::Context& context)
    : criteria(std::move(criteria_in)),
      initial_guess(std::move(initial_guess_in)),
      fast_flow(std::move(fast_flow_in)),  // NOLINT
      verbosity(std::move(verbosity_in)),  // NOLINT
      max_compute_coords_retries(max_compute_coords_retries_in),
      blocks_for_horizon_find(std::move(blocks_for_horizon_find_in)),
      rescaled_surface_char_speeds(std::move(rescaled_surface_char_speeds_in)) {
  if constexpr (not std::is_same_v<Fr, Frame::Distorted>) {
    if (rescaled_surface_char_speeds.has_value()) {
      PARSE_ERROR(context,
                  "RescaledSurfaceCharSpeeds requires the Distorted frame.");
    }
  }
}

template <typename Fr>
void HorizonOptions<Fr>::pup(PUP::er& p) {
  p | criteria;
  p | initial_guess;
  p | fast_flow;
  p | verbosity;
  p | max_compute_coords_retries;
  p | blocks_for_horizon_find;
  p | rescaled_surface_char_speeds;
}

template <typename Fr>
bool operator==(const HorizonOptions<Fr>& lhs, const HorizonOptions<Fr>& rhs) {
  if (lhs.criteria.size() != rhs.criteria.size()) {
    return false;
  }
  for (size_t i = 0; i < lhs.criteria.size(); ++i) {
    if (not(lhs.criteria[i]->is_equal(*(rhs.criteria[i])))) {
      return false;
    }
  }
  return lhs.initial_guess == rhs.initial_guess and
         lhs.fast_flow == rhs.fast_flow and lhs.verbosity == rhs.verbosity and
         lhs.max_compute_coords_retries == rhs.max_compute_coords_retries and
         lhs.blocks_for_horizon_find == rhs.blocks_for_horizon_find and
         lhs.rescaled_surface_char_speeds == rhs.rescaled_surface_char_speeds;
}

template <typename Fr>
bool operator!=(const HorizonOptions<Fr>& lhs, const HorizonOptions<Fr>& rhs) {
  return not(lhs == rhs);
}

// Explicit instantiations
#define FRAME(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                        \
  template struct HorizonOptions<FRAME(data)>;                      \
  template bool operator==(const HorizonOptions<FRAME(data)>& lhs,  \
                           const HorizonOptions<FRAME(data)>& rhs); \
  template bool operator!=(const HorizonOptions<FRAME(data)>& lhs,  \
                           const HorizonOptions<FRAME(data)>& rhs);
GENERATE_INSTANTIATIONS(INSTANTIATE,
                        (::Frame::Grid, ::Frame::Distorted, ::Frame::Inertial))

#undef FRAME
#undef INSTANTIATE
}  // namespace ah
