// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/TimeDependence/SphericalCompression.hpp"

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
#include "Domain/CoordinateMaps/MapInstantiationMacros.hpp"
#include "Domain/Creators/TimeDependence/TimeDependence.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace domain {
namespace creators::time_dependence {

template <size_t MeshDim>
SphericalCompression<MeshDim>::SphericalCompression(
    const double initial_time,
    const std::optional<double> initial_expiration_delta_t,
    const double min_radius, const double max_radius,
    const std::array<double, 3> center, const double initial_value,
    const double initial_velocity, const double initial_acceleration,
    std::string function_of_time_name) noexcept
    : initial_time_(initial_time),
      initial_expiration_delta_t_(initial_expiration_delta_t),
      min_radius_(min_radius),
      max_radius_(max_radius),
      center_(center),
      initial_value_(initial_value),
      initial_velocity_(initial_velocity),
      initial_acceleration_(initial_acceleration),
      function_of_time_name_(std::move(function_of_time_name)) {}

template <size_t MeshDim>
std::unique_ptr<TimeDependence<MeshDim>>
SphericalCompression<MeshDim>::get_clone() const noexcept {
  return std::make_unique<SphericalCompression>(
      initial_time_, initial_expiration_delta_t_, min_radius_, max_radius_,
      center_, initial_value_, initial_velocity_, initial_acceleration_,
      function_of_time_name_);
}

template <size_t MeshDim>
std::vector<std::unique_ptr<
    domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, MeshDim>>>
SphericalCompression<MeshDim>::block_maps(
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
SphericalCompression<MeshDim>::functions_of_time() const noexcept {
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      result{};
  result[function_of_time_name_] =
      std::make_unique<FunctionsOfTime::PiecewisePolynomial<3>>(
          initial_time_,
          std::array<DataVector, 4>{{{initial_value_},
                                     {initial_velocity_},
                                     {initial_acceleration_},
                                     {0.0}}},
          initial_expiration_delta_t_
              ? initial_time_ + *initial_expiration_delta_t_
              : std::numeric_limits<double>::max());
  return result;
}

template <size_t MeshDim>
auto SphericalCompression<MeshDim>::map_for_composition() const noexcept
    -> MapForComposition {
  return MapForComposition{SphericalCompressionMap{
      function_of_time_name_, min_radius_, max_radius_, center_}};
}

template <size_t Dim>
bool operator==(const SphericalCompression<Dim>& lhs,
                const SphericalCompression<Dim>& rhs) noexcept {
  return lhs.initial_time_ == rhs.initial_time_ and
         lhs.initial_expiration_delta_t_ == rhs.initial_expiration_delta_t_ and
         lhs.min_radius_ == rhs.min_radius_ and
         lhs.max_radius_ == rhs.max_radius_ and lhs.center_ == rhs.center_ and
         lhs.initial_value_ == rhs.initial_value_ and
         lhs.initial_velocity_ == rhs.initial_velocity_ and
         lhs.initial_acceleration_ == rhs.initial_acceleration_ and
         lhs.function_of_time_name_ == rhs.function_of_time_name_;
}

template <size_t Dim>
bool operator!=(const SphericalCompression<Dim>& lhs,
                const SphericalCompression<Dim>& rhs) noexcept {
  return not(lhs == rhs);
}

#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                              \
  template class SphericalCompression<GET_DIM(data)>;                       \
  template bool operator==                                                  \
      <GET_DIM(data)>(const SphericalCompression<GET_DIM(data)>&,           \
                      const SphericalCompression<GET_DIM(data)>&) noexcept; \
  template bool operator!=                                                  \
      <GET_DIM(data)>(const SphericalCompression<GET_DIM(data)>&,           \
                      const SphericalCompression<GET_DIM(data)>&) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATION, (3))

#undef GET_DIM
#undef INSTANTIATION
}  // namespace creators::time_dependence

using SphericalCompressionMap3d =
    CoordinateMaps::TimeDependent::SphericalCompression<false>;

INSTANTIATE_MAPS_FUNCTIONS(((SphericalCompressionMap3d)), (Frame::Grid),
                           (Frame::Inertial), (double, DataVector))

}  // namespace domain
