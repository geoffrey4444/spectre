// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>
#include <limits>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "PointwiseFunctions/SpecialRelativity/LorentzBoostMatrix.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"

namespace {
template <size_t SpatialDim, typename Frame>
void test_lorentz_boost_matrix_random(const double& used_for_size) noexcept {
  tnsr::aa<double, SpatialDim, Frame> (*f)(
      const tnsr::I<double, SpatialDim, Frame>&) =
      &sr::lorentz_boost_matrix<SpatialDim, Frame>;
  // The boost matrix is actually singular if the boost velocity is unity,
  // so let's ensure the speed never exceeds 0.999.
  constexpr double max_speed = 0.999;
  constexpr double upper = max_speed / static_cast<double>(SpatialDim);
  constexpr double lower = -upper;
  pypp::check_with_random_values<1>(f, "LorentzBoostMatrix",
                                    "lorentz_boost_matrix", {{{lower, upper}}},
                                    used_for_size);
}

template <size_t SpatialDim, typename Frame>
void test_lorentz_boost_matrix_analytic(
    const double& velocity_squared) noexcept {
  // Check that zero velocity returns an identity matrix
  const auto velocity_zero =
      make_with_value<tnsr::I<double, SpatialDim, Frame>>(velocity_squared,
                                                          0.0);
  const auto boost_matrix_zero = sr::lorentz_boost_matrix(velocity_zero);

  // Do not use DataStructures/Tensor/Identity.hpp, because identity returns
  // tnsr::Ab, whereas the boost matrix is returned as tnsr::aa.
  auto identity_matrix = make_with_value<tnsr::aa<double, SpatialDim, Frame>>(
      velocity_squared, 0.0);
  for (size_t i = 0; i < SpatialDim + 1; ++i) {
    identity_matrix.get(i, i) = 1.0;
  }
  CHECK_ITERABLE_APPROX(boost_matrix_zero, identity_matrix);

  // Check that the boost matrix inverse is the boost matrix with v->-v
  auto velocity = velocity_zero;
  for (size_t i = 0; i < SpatialDim; ++i) {
    velocity.get(i) =
        sqrt(velocity_squared) / sqrt(static_cast<double>(SpatialDim));
  }

  auto minus_velocity = velocity;
  for (size_t i = 0; i < SpatialDim; ++i) {
    minus_velocity.get(i) *= -1.0;
  }
  const auto boost_matrix = sr::lorentz_boost_matrix(velocity);
  const auto boost_matrix_minus = sr::lorentz_boost_matrix(minus_velocity);
  auto inverse_check = make_with_value<tnsr::aa<double, SpatialDim, Frame>>(
      velocity_squared, 0.0);
  for (size_t i = 0; i < SpatialDim + 1; ++i) {
    for (size_t j = i; j < SpatialDim + 1; ++j) {
      for (size_t k = 0; k < SpatialDim + 1; ++k) {
        inverse_check.get(i, j) +=
            boost_matrix.get(i, k) * boost_matrix_minus.get(k, j);
      }
    }
  }
  CHECK_ITERABLE_APPROX(inverse_check, identity_matrix);
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.SpecialRelativity.LorentzBoostMatrix",
    "[PointwiseFunctions][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env(
      "PointwiseFunctions/SpecialRelativity/");

  // CHECK_FOR_DOUBLES passes a double named d to the test function.
  // For test_lorentz_boost_matrix_random(), this double is only used for
  // size, so here set it to a signaling NaN.
  double d(std::numeric_limits<double>::signaling_NaN());
  CHECK_FOR_DOUBLES(test_lorentz_boost_matrix_random, (1, 2, 3),
                    (Frame::Grid, Frame::Inertial));

  const double small_velocity_squared = 5.0e-6;
  const double large_velocity_squared = 0.99;

  // The analytic test function uses the double that CHECK_FOR_DOUBLES passes to
  // set the magnitude of the velocity. Here, we test both for a small and for
  // a large velocity.
  d = small_velocity_squared;
  CHECK_FOR_DOUBLES(test_lorentz_boost_matrix_analytic, (1, 2, 3),
                    (Frame::Grid, Frame::Inertial));

  d = large_velocity_squared;
  CHECK_FOR_DOUBLES(test_lorentz_boost_matrix_analytic, (1, 2, 3),
                    (Frame::Grid, Frame::Inertial));
}
