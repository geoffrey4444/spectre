// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/TimeDependent/SphericalCompression.hpp"

#include <array>
#include <cmath>
#include <pup.h>
#include <pup_stl.h>
#include <string>
#include <type_traits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/StdHelpers.hpp"

namespace {
// Evaluate \f$\lambda_{00}(t) * Y_{00} = \lambda_{00}(t) / sqrt{4\pi}\f$.
double lambda00_y00(
    const std::string& f_of_t_name, const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) noexcept {
  ASSERT(functions_of_time.find(f_of_t_name) != functions_of_time.end(),
         "Could not find function of time: '"
             << f_of_t_name << "' in functions of time. Known functions are "
             << keys_of(functions_of_time));
  return functions_of_time.at(f_of_t_name)->func(time)[0][0] * 0.25 *
         M_2_SQRTPI;
}

// Evaluate \f$\lambda_{00}^{\prime}(t) / sqrt{4\pi}\f$.
double dt_lambda00_y00(
    const std::string& f_of_t_name, const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) noexcept {
  ASSERT(functions_of_time.find(f_of_t_name) != functions_of_time.end(),
         "Could not find function of time: '"
             << f_of_t_name << "' in functions of time. Known functions are "
             << keys_of(functions_of_time));
  return functions_of_time.at(f_of_t_name)->func_and_deriv(time)[1][0] * 0.25 *
         M_2_SQRTPI;
}

// Evaluate \f$\rho^i = \xi^i - C^i\f$ or \f$r^i = x^i - C^i\f$.
template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, 3> radial_position(
    const std::array<T, 3>& coords,
    const std::array<double, 3>& center) noexcept {
  std::array<tt::remove_cvref_wrap_t<T>, 3> result{};
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(result, i) = gsl::at(coords, i) - gsl::at(center, i);
  }
  return result;
}

// Adds the correction (terms other than the source coordinates) to
// compute either the mapped coordinate (if input is initially the source
// coordinates and and lambda_y is lambda00'(t) /sqrt(4*M_PI)) or the frame
// velocity (if input is initially zero and lambda_y is lambda00'(t)
// /sqrt(4*M_PI)).
template <typename T>
void correct_mapped_coordinate_or_frame_velocity(
    gsl::not_null<tt::remove_cvref_wrap_t<T>*> input,
    const T& source_radial_coord, const T& source_radius, const double lambda_y,
    const double min_radius, const double max_radius) noexcept {
  if (source_radius < min_radius) {
    *input -= lambda_y * source_radial_coord / min_radius;
  } else if (source_radius < max_radius) {
    *input -= lambda_y * source_radial_coord *
              (max_radius / source_radius - 1.0) / (max_radius - min_radius);
  }
}

// This function computes the (i,j) component of the Jacobian matrix
// \f$\partial x^i / \partial \xi^j.\f$
template <typename T>
void jacobian_component(gsl::not_null<tt::remove_cvref_wrap_t<T>*> input,
                        const std::array<T, 3>& source_radial_position,
                        const T& source_radius, const double lambda_y,
                        const double min_radius, const double max_radius,
                        const size_t i, const size_t j) noexcept {
  if (source_radius >= min_radius) {
    if (source_radius <= max_radius) {
      *input = gsl::at(source_radial_position, i) *
               gsl::at(source_radial_position, j) * lambda_y * (max_radius) /
               (max_radius - min_radius) / cube(source_radius);
      if (i == j) {
        *input += (1.0 - lambda_y * (max_radius / source_radius - 1.0) /
                             (max_radius - min_radius));
      }
    } else {
      if (i == j) {
        *input = 1.0;
      } else {
        *input = 0.0;
      }
    }
  } else {
    if (i == j) {
      *input = 1.0 - lambda_y / min_radius;
    } else {
      *input = 0.0;
    }
  }
}
}  // namespace

namespace domain::CoordinateMaps::TimeDependent {

SphericalCompression<3>::SphericalCompression(
    std::string function_of_time_name, const double min_radius,
    const double max_radius, const std::array<double, 3>& center) noexcept
    : f_of_t_name_(std::move(function_of_time_name)),
      min_radius_(min_radius),
      max_radius_(max_radius),
      center_(center) {}

template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, 3> SphericalCompression<3>::operator()(
    const std::array<T, 3>& source_coords, const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) const noexcept {
  const std::array<tt::remove_cvref_wrap_t<T>, 3> source_rad_position{
      radial_position(source_coords, center_)};
  const tt::remove_cvref_wrap_t<T> source_radius{
      magnitude(source_rad_position)};
  std::array<tt::remove_cvref_wrap_t<T>, 3> result{};
  for (size_t i = 0; i < 3; ++i) {
    // Initialize the mapped coordinates to the source coordinates.
    gsl::at(result, i) = gsl::at(source_coords, i);
    // result is either a double or a container (e.g. DataVector). Use
    // std::is_floating_point to check if T is double. If so, correct the
    // mapped coordinate (i.e., make the value in result equal to the value
    // of the mapped coordinate). If not, iterate over the values in the
    // container, correcting each one individually.
    if constexpr (std::is_floating_point<tt::remove_cvref_wrap_t<T>>::value) {
      correct_mapped_coordinate_or_frame_velocity(
          make_not_null(&gsl::at(result, i)), gsl::at(source_rad_position, i),
          source_radius, lambda00_y00(f_of_t_name_, time, functions_of_time),
          min_radius_, max_radius_);
    } else {
      const double lambda_y{
          lambda00_y00(f_of_t_name_, time, functions_of_time)};
      auto it_result = gsl::at(result, i).begin();
      auto it_radial_coord = gsl::at(source_rad_position, i).begin();
      auto it_radius = source_radius.begin();
      for (; it_result != gsl::at(result, i).end() and
             it_radial_coord != gsl::at(source_rad_position, i).end() and
             it_radius != source_radius.end();
           ++it_result, ++it_radial_coord, ++it_radius) {
        correct_mapped_coordinate_or_frame_velocity(
            make_not_null(&(*it_result)), *it_radial_coord, *it_radius,
            lambda_y, min_radius_, max_radius_);
      }
    }
  }
  return result;
}

std::optional<std::array<double, 3>> SphericalCompression<3>::inverse(
    const std::array<double, 3>& target_coords, const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) const noexcept {
  const double lambda_y{lambda00_y00(f_of_t_name_, time, functions_of_time)};
  if (UNLIKELY(min_radius_ - max_radius_ > lambda_y or
               lambda_y > min_radius_)) {
    return {};  // map not invertible
  } else {
    const std::array<double, 3> target_radial_position{
        radial_position(target_coords, center_)};
    const double target_radius{magnitude(target_radial_position)};
    if (target_radius > max_radius_) {
      return target_coords;
    } else {
      std::array<double, 3> result{target_radial_position};
      auto it_result = result.begin();
      auto it_center = center_.begin();
      for (; it_result != result.end() and it_center != center_.end();
           it_result++, it_center++) {
        if (target_radius < min_radius_ - lambda_y) {
          *it_result /= 1.0 - lambda_y / min_radius_;
        } else {
          *it_result *= 1.0 + lambda_y * max_radius_ /
                                  (max_radius_ - min_radius_) / target_radius;
          *it_result /= 1.0 + lambda_y / (max_radius_ - min_radius_);
        }
        *it_result += *it_center;
      }
      return {result};
    }
  }
}

template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, 3>
SphericalCompression<3>::frame_velocity(
    const std::array<T, 3>& source_coords, const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) const noexcept {
  const std::array<tt::remove_cvref_wrap_t<T>, 3> source_rad_position{
      radial_position(source_coords, center_)};
  const tt::remove_cvref_wrap_t<T> source_radius{
      magnitude(source_rad_position)};
  std::array<tt::remove_cvref_wrap_t<T>, 3> result{};
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(result, i) = 0.0;
    // result is either a double or a container (e.g. DataVector). Use
    // std::is_floating_point to check if T is double. If so, correct the
    // mapped coordinate (i.e., make the value in result equal to the value
    // of the mapped coordinate). If not, iterate over the values in the
    // container, correcting each one individually.
    if constexpr (std::is_floating_point<tt::remove_cvref_wrap_t<T>>::value) {
      correct_mapped_coordinate_or_frame_velocity(
          make_not_null(&gsl::at(result, i)), gsl::at(source_rad_position, i),
          source_radius, dt_lambda00_y00(f_of_t_name_, time, functions_of_time),
          min_radius_, max_radius_);
    } else {
      const double lambda_y{
          dt_lambda00_y00(f_of_t_name_, time, functions_of_time)};
      auto it_result = gsl::at(result, i).begin();
      auto it_radial_coord = gsl::at(source_rad_position, i).begin();
      auto it_radius = source_radius.begin();
      for (; it_result != gsl::at(result, i).end() and
             it_radial_coord != gsl::at(source_rad_position, i).end() and
             it_radius != source_radius.end();
           ++it_result, ++it_radial_coord, ++it_radius) {
        correct_mapped_coordinate_or_frame_velocity(
            make_not_null(&(*it_result)), *it_radial_coord, *it_radius,
            lambda_y, min_radius_, max_radius_);
      }
    }
  }
  return result;
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>
SphericalCompression<3>::jacobian(
    const std::array<T, 3>& source_coords, const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) const noexcept {
  const std::array<tt::remove_cvref_wrap_t<T>, 3> source_rad_position{
      radial_position(source_coords, center_)};
  const tt::remove_cvref_wrap_t<T> source_radius{
      magnitude(source_rad_position)};
  const double lambda_y{lambda00_y00(f_of_t_name_, time, functions_of_time)};
  auto jacobian_matrix{
      make_with_value<tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>>(
          dereference_wrapper(source_coords[0]),
          std::numeric_limits<double>::signaling_NaN())};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      // result is either a double or a container (e.g. DataVector). Use
      // std::is_floating_point to check if T is double. If so, correct the
      // mapped coordinate (i.e., make the value in result equal to the value
      // of the mapped coordinate). If not, iterate over the values in the
      // container, correcting each one individually.
      if constexpr (std::is_floating_point<tt::remove_cvref_wrap_t<T>>::value) {
        jacobian_component(make_not_null(&(jacobian_matrix.get(i, j))),
                           source_rad_position, source_radius, lambda_y,
                           min_radius_, max_radius_, i, j);
      } else {
        auto it_result = jacobian_matrix.get(i, j).begin();
        auto it_radial_x = source_rad_position[0].begin();
        auto it_radial_y = source_rad_position[1].begin();
        auto it_radial_z = source_rad_position[2].begin();
        auto it_radius = source_radius.begin();
        std::array<double, 3> source_point_rad_position{};
        for (; it_result != jacobian_matrix.get(i, j).end() and
               it_radial_x != source_rad_position[0].end() and
               it_radial_y != source_rad_position[1].end() and
               it_radial_z != source_rad_position[2].end() and
               it_radius != source_radius.end();
             ++it_result, ++it_radial_x, ++it_radial_y, ++it_radial_z,
             ++it_radius) {
          source_point_rad_position[0] = *it_radial_x;
          source_point_rad_position[1] = *it_radial_y;
          source_point_rad_position[2] = *it_radial_z;
          jacobian_component(make_not_null(&(*it_result)),
                             source_point_rad_position, *it_radius, lambda_y,
                             min_radius_, max_radius_, i, j);
        }
      }
    }
  }
  return jacobian_matrix;
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>
SphericalCompression<3>::inv_jacobian(
    const std::array<T, 3>& source_coords, const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) const noexcept {
  // Obtain the Jacobian and then compute its inverse numerically
  return determinant_and_inverse(
             jacobian(source_coords, time, functions_of_time))
      .second;
}

void SphericalCompression<3>::pup(PUP::er& p) noexcept {
  p | f_of_t_name_;
  p | min_radius_;
  p | max_radius_;
  p | center_;
}

bool operator==(const SphericalCompression<3>& lhs,
                const SphericalCompression<3>& rhs) noexcept {
  return lhs.f_of_t_name_ == rhs.f_of_t_name_ and
         lhs.min_radius_ == rhs.min_radius_ and
         lhs.max_radius_ == rhs.max_radius_ and lhs.center_ == rhs.center_;
}

bool operator!=(const SphericalCompression<3>& lhs,
                const SphericalCompression<3>& rhs) noexcept {
  return not(lhs == rhs);
}

/// \cond
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                                \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, DIM(data)>      \
  SphericalCompression<DIM(data)>::operator()(                              \
      const std::array<DTYPE(data), DIM(data)>& source_coords, double time, \
      const std::unordered_map<                                             \
          std::string,                                                      \
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&        \
          functions_of_time) const noexcept;                                \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, DIM(data)>      \
  SphericalCompression<DIM(data)>::frame_velocity(                          \
      const std::array<DTYPE(data), DIM(data)>& source_coords,              \
      const double time,                                                    \
      const std::unordered_map<                                             \
          std::string,                                                      \
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&        \
          functions_of_time) const noexcept;                                \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, DIM(data),        \
                    Frame::NoFrame>                                         \
  SphericalCompression<DIM(data)>::jacobian(                                \
      const std::array<DTYPE(data), DIM(data)>& source_coords, double time, \
      const std::unordered_map<                                             \
          std::string,                                                      \
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&        \
          functions_of_time) const noexcept;                                \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, DIM(data),        \
                    Frame::NoFrame>                                         \
  SphericalCompression<DIM(data)>::inv_jacobian(                            \
      const std::array<DTYPE(data), DIM(data)>& source_coords, double time, \
      const std::unordered_map<                                             \
          std::string,                                                      \
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&        \
          functions_of_time) const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (3),
                        (double, DataVector,
                         std::reference_wrapper<const double>,
                         std::reference_wrapper<const DataVector>))
#undef DIM
#undef DTYPE
#undef INSTANTIATE
/// \endcond

}  // namespace domain::CoordinateMaps::TimeDependent
