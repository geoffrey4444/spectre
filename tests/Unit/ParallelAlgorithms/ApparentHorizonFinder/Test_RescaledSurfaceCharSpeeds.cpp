// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/Strahlkorper/Strahlkorper.hpp"
#include "NumericalAlgorithms/Strahlkorper/StrahlkorperFunctions.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/RescaledSurfaceCharSpeeds.hpp"
#include "Utilities/Gsl.hpp"

namespace {
void test_factors() {
  const DataVector horizon_radius{2.0, 3.0, 4.0};
  const DataVector excision_radius{1.0, 2.4, 2.0};
  const auto factors =
      ah::rescaled_surface_factors(horizon_radius, excision_radius, 5, 0.01);
  REQUIRE(factors.has_value());
  CHECK_ITERABLE_APPROX(*factors,
                        (std::vector<double>{1.0, 0.988, 0.952, 0.892, 0.808}));
  const auto two_factors =
      ah::rescaled_surface_factors(horizon_radius, excision_radius, 2, 0.0);
  REQUIRE(two_factors.has_value());
  CHECK_ITERABLE_APPROX(*two_factors, (std::vector<double>{1.0, 0.8}));
  const auto scaled_factors = ah::rescaled_surface_factors(
      3.0 * horizon_radius, 3.0 * excision_radius, 5, 0.01);
  REQUIRE(scaled_factors.has_value());
  CHECK_ITERABLE_APPROX(*factors, *scaled_factors);
  for (const auto factor : *factors) {
    for (size_t p = 0; p < horizon_radius.size(); ++p) {
      CHECK(factor * horizon_radius[p] > excision_radius[p]);
    }
  }
  CHECK_FALSE(
      ah::rescaled_surface_factors(horizon_radius, horizon_radius, 5, 0.0)
          .has_value());
  CHECK_FALSE(
      ah::rescaled_surface_factors(horizon_radius, 1.1 * horizon_radius, 5, 0.0)
          .has_value());
  CHECK_FALSE(
      ah::rescaled_surface_factors(horizon_radius, excision_radius, 5, 0.3)
          .has_value());
  CHECK_FALSE(
      ah::rescaled_surface_factors(horizon_radius, excision_radius, 1, 0.0)
          .has_value());
  CHECK_FALSE(
      ah::rescaled_surface_factors(horizon_radius, excision_radius, 5, -0.1)
          .has_value());
  CHECK_FALSE(ah::rescaled_surface_factors({}, {}, 5, 0.0).has_value());
  CHECK_FALSE(
      ah::rescaled_surface_factors(horizon_radius, {1.0}, 5, 0.0).has_value());
  for (const double invalid :
       {0.0, -1.0, std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::quiet_NaN()}) {
    CHECK_FALSE(ah::rescaled_surface_factors({2.0, invalid}, {1.0, 1.0}, 5, 0.0)
                    .has_value());
    CHECK_FALSE(ah::rescaled_surface_factors({2.0, 2.0}, {1.0, invalid}, 5, 0.0)
                    .has_value());
  }
  CHECK_FALSE(
      ah::rescaled_surface_factors(horizon_radius, excision_radius, 5,
                                   std::numeric_limits<double>::quiet_NaN())
          .has_value());
  CHECK_FALSE(
      ah::rescaled_surface_factors(horizon_radius, excision_radius, 5,
                                   std::numeric_limits<double>::infinity())
          .has_value());
}

void test_schwarzschild() {
  using Fr = Frame::Distorted;
  for (const double radius : {1.5, 2.0, 2.5}) {
    const ylm::Strahlkorper<Fr> surface{4, radius, {{0.0, 0.0, 0.0}}};
    const auto rhat = ylm::rhat(ylm::theta_phi(surface));
    const auto number_of_points = get<0>(rhat).size();
    const double twice_mass_over_radius = 2.0 / radius;
    const double metric_radial = 1.0 + twice_mass_over_radius;
    const Scalar<DataVector> lapse{number_of_points, 1.0 / sqrt(metric_radial)};
    tnsr::I<DataVector, 3, Fr> shift{number_of_points};
    tnsr::II<DataVector, 3, Fr> inverse_metric{number_of_points};
    for (size_t i = 0; i < 3; ++i) {
      shift.get(i) = twice_mass_over_radius / metric_radial * rhat.get(i);
      for (size_t j = i; j < 3; ++j) {
        inverse_metric.get(i, j) =
            (i == j ? 1.0 : 0.0) -
            twice_mass_over_radius / metric_radial * rhat.get(i) * rhat.get(j);
      }
    }
    for (const double radial_velocity : {0.0, -0.1, 0.2}) {
      auto time_deriv_surface = surface;
      time_deriv_surface.coefficients() *= radial_velocity / radius;
      const auto extrema = ah::rescaled_surface_char_speed_extrema(
          surface, time_deriv_surface, lapse, shift, inverse_metric);
      const double expected =
          (twice_mass_over_radius - 1.0) / sqrt(metric_radial) +
          sqrt(metric_radial) * radial_velocity;
      CHECK(extrema.first == approx(expected));
      CHECK(extrema.second == approx(expected));
    }
  }
}

void test_deformed_surface() {
  using Fr = Frame::Distorted;
  const ylm::Strahlkorper<Fr> sphere{5, 2.0, {{0.0, 0.0, 0.0}}};
  const auto rhat = ylm::rhat(ylm::theta_phi(sphere));
  const DataVector radius = 2.0 + 0.3 * get<2>(rhat);
  const DataVector radial_velocity = 0.1 + 0.05 * get<2>(rhat);
  const ylm::Strahlkorper<Fr> surface{5, 5, radius, {{0.0, 0.0, 0.0}}};
  const ylm::Strahlkorper<Fr> time_deriv_surface{
      5, 5, radial_velocity, {{0.0, 0.0, 0.0}}};
  const auto number_of_points = radius.size();
  Scalar<DataVector> lapse{number_of_points};
  get(lapse) = 1.0 + 0.2 * get<2>(rhat);
  tnsr::I<DataVector, 3, Fr> shift{number_of_points, 0.0};
  get<0>(shift) = 0.2;
  get<1>(shift) = -0.4;
  get<2>(shift) = 0.7;
  tnsr::II<DataVector, 3, Fr> inverse_metric{number_of_points, 0.0};
  get<0, 0>(inverse_metric) = 1.4;
  get<1, 1>(inverse_metric) = 0.8;
  get<2, 2>(inverse_metric) = 1.1;
  std::vector<double> expected_speeds(number_of_points);
  // For R = 2 + 0.3 cos(theta), the Cartesian normal is analytically
  // rhat - (0.3 / R) (ez - cos(theta) rhat).
  for (size_t p = 0; p < number_of_points; ++p) {
    const double cosine = get<2>(rhat)[p];
    const std::array<double, 3> normal{
        get<0>(rhat)[p] * (1.0 + 0.3 * cosine / radius[p]),
        get<1>(rhat)[p] * (1.0 + 0.3 * cosine / radius[p]),
        cosine - 0.3 / radius[p] * (1.0 - cosine * cosine)};
    const double normal_magnitude =
        sqrt(1.4 * normal[0] * normal[0] + 0.8 * normal[1] * normal[1] +
             1.1 * normal[2] * normal[2]);
    expected_speeds[p] = -get(lapse)[p];
    for (size_t i = 0; i < 3; ++i) {
      expected_speeds[p] +=
          gsl::at(normal, i) / normal_magnitude *
          (shift.get(i)[p] + radial_velocity[p] * rhat.get(i)[p]);
    }
  }
  const auto expected_extrema =
      std::minmax_element(expected_speeds.begin(), expected_speeds.end());
  const auto extrema = ah::rescaled_surface_char_speed_extrema(
      surface, time_deriv_surface, lapse, shift, inverse_metric);
  CHECK(extrema.first == approx(*expected_extrema.first));
  CHECK(extrema.second == approx(*expected_extrema.second));
  CHECK(extrema.first < extrema.second);

  // Moving a radial contribution from the surface velocity to the coordinate
  // shift must leave the measured speed unchanged.
  auto shifted_shift = shift;
  for (size_t i = 0; i < 3; ++i) {
    shifted_shift.get(i) += 0.3 * rhat.get(i);
  }
  const ylm::Strahlkorper<Fr> shifted_time_deriv_surface{
      5, 5, radial_velocity - 0.3, {{0.0, 0.0, 0.0}}};
  const auto shifted_extrema = ah::rescaled_surface_char_speed_extrema(
      surface, shifted_time_deriv_surface, lapse, shifted_shift,
      inverse_metric);
  CHECK(shifted_extrema.first == approx(extrema.first));
  CHECK(shifted_extrema.second == approx(extrema.second));

  for (const double invalid : {std::numeric_limits<double>::infinity(),
                               std::numeric_limits<double>::quiet_NaN()}) {
    get(lapse)[number_of_points / 2] = invalid;
    const auto invalid_extrema = ah::rescaled_surface_char_speed_extrema(
        surface, time_deriv_surface, lapse, shift, inverse_metric);
    CHECK(std::isnan(invalid_extrema.first));
    CHECK(std::isnan(invalid_extrema.second));
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.ApparentHorizonFinder.RescaledSurfaceCharSpeeds",
                  "[Unit][ApparentHorizonFinder]") {
  test_factors();
  test_schwarzschild();
  test_deformed_surface();
}
