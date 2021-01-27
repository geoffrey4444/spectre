// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/TimeDependence/BbhSphericalCompression.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/MapInstantiationMacros.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ProductMaps.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ProductMaps.tpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace domain {
namespace creators::time_dependence {

template <size_t MeshDim>
BbhSphericalCompression<MeshDim>::BbhSphericalCompression(
    double initial_time, std::optional<double> initial_expiration_delta_t,
    double compression_velocity_a, compression_velocity_b,
    double inner_radius_object_a, double outer_radius_object_a,
    double x_coord_object_a, double inner_radius_object_b,
    double outer_radius_object_b, double x_coord_object_b,
    std::string function_of_time_a, std::string function_of_time_b) noexcept
    : initial_time_(initial_time),
      initial_expiration_delta_t_(initial_expiration_delta_t),
      compression_velocity_a_(compression_velocity_a),
      compression_velocity_b_(compression_velocity_b),
      inner_radius_object_a_(inner_radius_object_a),
      outer_radius_object_a_(outer_radius_object_a),
      x_coord_object_a_(x_coord_object_a),
      inner_radius_object_b_(inner_radius_object_b),
      outer_radius_object_b_(outer_radius_object_b),
      x_coord_object_b_(x_coord_object_b),
      function_of_time_a_(std::move(function_of_time_name)),
      function_of_time_b_(std::move(function_of_time_b)) {}

template <size_t MeshDim>
std::unique_ptr<TimeDependence<MeshDim>>
BbhSphericalCompression<MeshDim>::get_clone() const noexcept {
  return std::make_unique<BbhSphericalCompression>(
      initial_time_, initial_expiration_delta_t_, compression_velocity_a_,
      compression_velocity_b_, inner_radius_object_a_, outer_radius_object_a_,
      x_coord_object_a_, inner_radius_object_b_, outer_radius_object_b_,
      x_coord_object_b_, std::string function_of_time_a_,
      std::string function_of_time_b_);
}

template <size_t MeshDim>
std::vector<std::unique_ptr<
    domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, MeshDim>>>
BbhSphericalCompression<MeshDim>::block_maps(
    const size_t number_of_blocks) const noexcept {
  ASSERT(number_of_blocks > 0, "Must have at least one block to create.");
  std::vector<std::unique_ptr<
      domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, MeshDim>>>
      result{number_of_blocks};
  result[0] = std::make_unique<MapForComposition>(map_for_composition());
  for (size_t i = 1; i < number_of_blocks; ++i) {
    result[i] = result[0]->get_clone();
  }
  return result;
}

template <size_t MeshDim>
std::unordered_map<std::string,
                   std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
BbhSphericalCompression<MeshDim>::functions_of_time() const noexcept {
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      result{};
  // We use a third-order `PiecewisePolynomial` to ensure sufficiently
  // smooth behavior of the function of time
  result[function_of_time_a_] =
      std::make_unique<FunctionsOfTime::PiecewisePolynomial<3>>(
          initial_time_,
          std::array<DataVector, 4>{
              {{0.0}, {compression_velocity_a_}, {0.0}, {0.0}}},
          initial_expiration_delta_t_
              ? initial_time_ + *initial_expiration_delta_t_
              : std::numeric_limits<double>::max());
  result[function_of_time_b_] =
      std::make_unique<FunctionsOfTime::PiecewisePolynomial<3>>(
          initial_time_,
          std::array<DataVector, 4>{
              {{0.0}, {compression_velocity_b_}, {0.0}, {0.0}}},
          initial_expiration_delta_t_
              ? initial_time_ + *initial_expiration_delta_t_
              : std::numeric_limits<double>::max());
  return result;
}

/// \cond
template <>
auto BbhSphericalCompression<3>::map_for_composition() const noexcept
    -> MapForComposition {
  using ProductMap = domain::CoordinateMaps::TimeDependent::ProductOf2Maps<
      domain::CoordinateMaps::TimeDependent::Rotation<2>,
      domain::CoordinateMaps::Identity<1>>;
  return MapForComposition{
      ProductMap{domain::CoordinateMaps::TimeDependent::Rotation<2>{
                     function_of_time_name_},
                 domain::CoordinateMaps::Identity<1>{}}};
}

template <size_t Dim>
bool operator==(const BbhSphericalCompression<Dim>& lhs,
                const BbhSphericalCompression<Dim>& rhs) noexcept {
  return lhs.initial_time_ == rhs.initial_time_ and
         lhs.initial_expiration_delta_t_ == rhs.initial_expiration_delta_t_ and
         lhs.compression_velocity_a_ == rhs.compression_velocity_a_ and
         lhs.compression_velocity_b_ == rhs.compression_velocity_b_ and
         lhs.inner_radius_object_a_ == rhs.inner_radius_object_a_ and
         lhs.outer_radius_object_a_ == rhs.outer_radius_object_a_ and
         lhs.x_coord_object_a_ == rhs.x_coord_object_a_ and
         lhs.inner_radius_object_b_ == rhs.inner_radius_object_b_ and
         lhs.outer_radius_object_b_ == rhs.outer_radius_object_b_ and
         lhs.x_coord_object_b_ == rhs.x_coord_object_b_ and
         lhs.function_of_time_a_ == rhs.function_of_time_a_ and
         lhs.function_of_time_b_ == rhs.function_of_time_b_;
}

template <size_t Dim>
bool operator!=(const BbhSphericalCompression<Dim>& lhs,
                const BbhSphericalCompression<Dim>& rhs) noexcept {
  return not(lhs == rhs);
}

#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                                 \
  template class BbhSphericalCompression<GET_DIM(data)>;                       \
  template bool operator==                                                     \
      <GET_DIM(data)>(const BbhSphericalCompression<GET_DIM(data)>&,           \
                      const BbhSphericalCompression<GET_DIM(data)>&) noexcept; \
  template bool operator!=                                                     \
      <GET_DIM(data)>(const BbhSphericalCompression<GET_DIM(data)>&,           \
                      const BbhSphericalCompression<GET_DIM(data)>&) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATION, (2, 3))

#undef GET_DIM
#undef INSTANTIATION
/// \endcond
}  // namespace creators::time_dependence

using Identity = CoordinateMaps::Identity<1>;
using Rotation2d = CoordinateMaps::TimeDependent::Rotation<2>;
using Rotation3d =
    CoordinateMaps::TimeDependent::ProductOf2Maps<Rotation2d, Identity>;

template class CoordinateMaps::TimeDependent::ProductOf2Maps<Rotation2d,
                                                             Identity>;

INSTANTIATE_MAPS_FUNCTIONS(((Rotation2d), (Rotation3d)), (Frame::Grid),
                           (Frame::Inertial), (double, DataVector))

}  // namespace domain
