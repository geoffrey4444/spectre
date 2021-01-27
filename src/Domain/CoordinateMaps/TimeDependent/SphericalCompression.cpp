// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/TimeDependent/SphericalCompression.hpp"

#include <array>
#include <cmath>
#include <iomanip>
#include <limits>
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
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/StdHelpers.hpp"

namespace {
template <typename T>
using ResultType = tt::remove_cvref_wrap_t<T>;

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
std::array<ResultType<T>, 3> radial_position(
    const std::array<T, 3>& coords,
    const std::array<double, 3>& center) noexcept {
  std::array<ResultType<T>, 3> result{};
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
template <bool InteriorMap, typename T>
void correct_mapped_coordinate_or_frame_velocity(
    gsl::not_null<ResultType<T>*> input, const T& source_radial_coord,
    const T& source_radius, const double lambda_y, const double min_radius,
    const double max_radius) noexcept {
  if constexpr (InteriorMap) {
    *input -= lambda_y * source_radial_coord / min_radius;
  } else {
    *input -= lambda_y * source_radial_coord *
              (max_radius / source_radius - 1.0) / (max_radius - min_radius);
  }
}

// This function computes the (i,j) component of the Jacobian matrix
// \f$\partial x^i / \partial \xi^j.\f$
template <bool InteriorMap, typename T>
void jacobian_component(gsl::not_null<ResultType<T>*> input,
                        const std::array<T, 3>& source_radial_position,
                        const T& source_radius, const double lambda_y,
                        const double min_radius, const double max_radius,
                        const size_t i, const size_t j) noexcept {
  if constexpr (not InteriorMap) {
    *input = gsl::at(source_radial_position, i) *
             gsl::at(source_radial_position, j) * lambda_y * (max_radius) /
             (max_radius - min_radius) / cube(source_radius);
    if (i == j) {
      *input += (1.0 - lambda_y * (max_radius / source_radius - 1.0) /
                           (max_radius - min_radius));
    }
  } else {
    if (i == j) {
      *input = 1.0 - lambda_y / min_radius;
    } else {
      *input = 0.0;
    }
  }
}

template <bool InteriorMap, typename T>
void check_source_radius(const T& radius, const double min_radius,
                         const double max_radius) {
  auto radius_in_expected_range = [](const T& source_radius,
                                     const double min_rad,
                                     const double max_rad) {
    if constexpr (std::is_floating_point<ResultType<T>>::value) {
      if constexpr (InteriorMap) {
        return source_radius <=
               min_rad + std::numeric_limits<double>::epsilon();
      } else {
        return source_radius >=
                   min_rad - std::numeric_limits<double>::epsilon() and
               source_radius <=
                   max_rad + std::numeric_limits<double>::epsilon();
      }
    } else {
      if constexpr (InteriorMap) {
        return max(source_radius) <=
               min_rad + std::numeric_limits<double>::epsilon();
      } else {
        return min(source_radius) >=
                   min_rad - std::numeric_limits<double>::epsilon() and
               max(source_radius) <=
                   max_rad + std::numeric_limits<double>::epsilon();
      }
    }
  };
  if (!radius_in_expected_range(radius, min_radius, max_radius)) {
    ERROR("Source radius " << std::setprecision(20) << radius
                           << " not in expected range. min_radius == "
                           << min_radius << ", max_radius == " << max_radius
                           << ", InteriorMap == " << InteriorMap << "\n");
  }
}
}  // namespace

namespace domain::CoordinateMaps::TimeDependent {
template <bool InteriorMap>
SphericalCompression<InteriorMap>::SphericalCompression(
    std::string function_of_time_name, const double min_radius,
    const double max_radius, const std::array<double, 3>& center) noexcept
    : f_of_t_name_(std::move(function_of_time_name)),
      min_radius_(min_radius),
      max_radius_(max_radius),
      center_(center) {
  ASSERT(max_radius - min_radius > 1.0e-10,
         "max_radius must be greater than min_radius, but max_radius - "
         "min_radius == "
             << max_radius - min_radius);
}

template <bool InteriorMap>
template <typename T>
std::array<ResultType<T>, 3> SphericalCompression<InteriorMap>::operator()(
    const std::array<T, 3>& source_coords, const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) const noexcept {
  const std::array<ResultType<T>, 3> source_rad_position{
      radial_position(source_coords, center_)};
  const ResultType<T> source_radius{magnitude(source_rad_position)};
  check_source_radius<InteriorMap>(source_radius, min_radius_, max_radius_);
  std::array<ResultType<T>, 3> result{};
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(result, i) = gsl::at(source_coords, i);
    correct_mapped_coordinate_or_frame_velocity<InteriorMap>(
        make_not_null(&gsl::at(result, i)), gsl::at(source_rad_position, i),
        source_radius, lambda00_y00(f_of_t_name_, time, functions_of_time),
        min_radius_, max_radius_);
  }
  return result;
}

template <bool InteriorMap>
std::optional<std::array<double, 3>> SphericalCompression<InteriorMap>::inverse(
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
    if constexpr (InteriorMap) {
      if (!(target_radius <= min_radius_ - lambda_y)) {
        ERROR("Target radius "
              << target_radius
              << " must be less than (min_radius_ - lambda_y == "
              << min_radius_ - lambda_y << ") "
              << "when InteriorMap is true in SphericalCompression CoordMap");
      }
    } else {
      if (!((target_radius >= min_radius_ - lambda_y) and
            target_radius <= max_radius_)) {
        ERROR("Target radius "
              << target_radius << " must be between (min_radius_ - lambda_y) "
              << min_radius_ - lambda_y << " and max_radius_ " << max_radius_
              << " (inclusive) when InteriorMap is false in "
                 "SphericalCompression CoordMap");
      }
    }
    double scale{};
    if constexpr (InteriorMap) {
      scale = 1.0 / (1.0 - lambda_y / min_radius_);
    } else {
      scale = (1.0 + lambda_y * max_radius_ / (max_radius_ - min_radius_) /
                         target_radius) /
              (1.0 + lambda_y / (max_radius_ - min_radius_));
    }
    std::array<double, 3> result{target_radial_position};
    result[0] = result[0] * scale + center_[0];
    result[1] = result[1] * scale + center_[1];
    result[2] = result[2] * scale + center_[2];
    return {result};
  }
}

template <bool InteriorMap>
template <typename T>
std::array<ResultType<T>, 3> SphericalCompression<InteriorMap>::frame_velocity(
    const std::array<T, 3>& source_coords, const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) const noexcept {
  const std::array<ResultType<T>, 3> source_rad_position{
      radial_position(source_coords, center_)};
  const ResultType<T> source_radius{magnitude(source_rad_position)};
  check_source_radius<InteriorMap>(source_radius, min_radius_, max_radius_);
  std::array<ResultType<T>, 3> result{};
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(result, i) = 0.0;
    correct_mapped_coordinate_or_frame_velocity<InteriorMap>(
        make_not_null(&gsl::at(result, i)), gsl::at(source_rad_position, i),
        source_radius, dt_lambda00_y00(f_of_t_name_, time, functions_of_time),
        min_radius_, max_radius_);
  }
  return result;
}

template <bool InteriorMap>
template <typename T>
tnsr::Ij<ResultType<T>, 3, Frame::NoFrame>
SphericalCompression<InteriorMap>::jacobian(
    const std::array<T, 3>& source_coords, const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) const noexcept {
  const std::array<ResultType<T>, 3> source_rad_position{
      radial_position(source_coords, center_)};
  const ResultType<T> source_radius{magnitude(source_rad_position)};
  const double lambda_y{lambda00_y00(f_of_t_name_, time, functions_of_time)};
  check_source_radius<InteriorMap>(source_radius, min_radius_, max_radius_);
  auto jacobian_matrix{
      make_with_value<tnsr::Ij<ResultType<T>, 3, Frame::NoFrame>>(
          dereference_wrapper(source_coords[0]),
          std::numeric_limits<double>::signaling_NaN())};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
        jacobian_component<InteriorMap>(
            make_not_null(&(jacobian_matrix.get(i, j))), source_rad_position,
            source_radius, lambda_y, min_radius_, max_radius_, i, j);
    }
  }
  return jacobian_matrix;
}

template <bool InteriorMap>
template <typename T>
tnsr::Ij<ResultType<T>, 3, Frame::NoFrame>
SphericalCompression<InteriorMap>::inv_jacobian(
    const std::array<T, 3>& source_coords, const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) const noexcept {
  // Obtain the Jacobian and then compute its inverse numerically
  return determinant_and_inverse(
             jacobian(source_coords, time, functions_of_time))
      .second;
}

template <bool InteriorMap>
void SphericalCompression<InteriorMap>::pup(PUP::er& p) noexcept {
  p | f_of_t_name_;
  p | min_radius_;
  p | max_radius_;
  p | center_;
}

/// \cond
#define INTERIOR_MAP(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                                 \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 3>               \
  SphericalCompression<INTERIOR_MAP(data)>::operator()(                      \
      const std::array<DTYPE(data), 3>& source_coords, double time,          \
      const std::unordered_map<                                              \
          std::string,                                                       \
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&         \
          functions_of_time) const noexcept;                                 \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 3>               \
  SphericalCompression<INTERIOR_MAP(data)>::frame_velocity(                  \
      const std::array<DTYPE(data), 3>& source_coords, const double time,    \
      const std::unordered_map<                                              \
          std::string,                                                       \
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&         \
          functions_of_time) const noexcept;                                 \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 3, Frame::NoFrame> \
  SphericalCompression<INTERIOR_MAP(data)>::jacobian(                        \
      const std::array<DTYPE(data), 3>& source_coords, double time,          \
      const std::unordered_map<                                              \
          std::string,                                                       \
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&         \
          functions_of_time) const noexcept;                                 \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 3, Frame::NoFrame> \
  SphericalCompression<INTERIOR_MAP(data)>::inv_jacobian(                    \
      const std::array<DTYPE(data), 3>& source_coords, double time,          \
      const std::unordered_map<                                              \
          std::string,                                                       \
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&         \
          functions_of_time) const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (true, false),
                        (double, DataVector,
                         std::reference_wrapper<const double>,
                         std::reference_wrapper<const DataVector>))
#undef INTERIOR_MAP
#undef DTYPE
#undef INSTANTIATE

#define INTERIOR_MAP(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATE(_, data)                                                  \
  template SphericalCompression<INTERIOR_MAP(data)>::SphericalCompression(    \
      std::string function_of_time_name, const double min_radius,             \
      const double max_radius, const std::array<double, 3>& center) noexcept; \
  template void SphericalCompression<INTERIOR_MAP(data)>::pup(                \
      PUP::er& p) noexcept;                                                   \
  template std::optional<std::array<double, 3>>                               \
  SphericalCompression<INTERIOR_MAP(data)>::inverse(                          \
      const std::array<double, 3>& target_coords, const double time,          \
      const std::unordered_map<                                               \
          std::string,                                                        \
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&          \
          functions_of_time) const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (true, false))
#undef INTERIOR_MAP
#undef INSTANTIATE

/// \endcond

}  // namespace domain::CoordinateMaps::TimeDependent