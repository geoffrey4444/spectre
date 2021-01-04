// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <pup.h>
#include <random>
#include <string>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/TimeDependent/SphericalCompression.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "ErrorHandling/Error.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Domain/CoordinateMaps/TestMapHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/StdArrayHelpers.hpp"

namespace {
// Generates the map, time, an a FunctionOfTime
void generate_map_time_and_f_of_time(
    gsl::not_null<
        domain::CoordinateMaps::TimeDependent::SphericalCompression<3>*>
        map,
    gsl::not_null<double*> time,
    gsl::not_null<std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>*>
        functions_of_time,
    gsl::not_null<double*> min_radius, gsl::not_null<double*> max_radius,
    gsl::not_null<std::array<double, 3>*> center,
    gsl::not_null<std::mt19937*> generator) noexcept {
  // Set up the map
  const std::string f_of_t_name{"ExpansionFactor"};
  std::uniform_real_distribution<> rad_dis{0.4, 0.7};
  std::uniform_real_distribution<> drad_dis{0.1, 0.2};
  const double rad{rad_dis(*generator)};
  *min_radius = rad - drad_dis(*generator);
  *max_radius = rad + drad_dis(*generator);
  std::uniform_real_distribution<> center_dis{-0.05, 0.05};
  *center = std::array<double, 3>{
      {center_dis(*generator), center_dis(*generator), center_dis(*generator)}};
  domain::CoordinateMaps::TimeDependent::SphericalCompression<3> random_map{
      f_of_t_name, *min_radius, *max_radius, *center};
  *map = random_map;

  // Choose a random time for evaluating the FunctionOfTime
  std::uniform_real_distribution<> time_dis{-1.0, 1.0};
  *time = time_dis(*generator);

  // Create a FunctionsOfTime containing a FunctionOfTime that, when evaluated
  // at time, is invertible. Recall that the map is invertible if
  // min_radius - max_radius < lambda(t) / sqrt(4*pi) < min_radius. So
  // lambda(t) = (min_radius - eps * max_radius) * sqrt(4*pi) is guaranteed
  // invertible for 0 < eps < 1. So set the constant term in the piecewise
  // polynomial to a0 = (min_radius - eps * max_radius) * sqrt(4*pi).
  // The remaining coefficients a1, a2, a3 will be chosen as a random
  // small factor of a0, to ensure that these terms don't make the map
  // non-invertible.
  std::uniform_real_distribution<> eps_dis{0.2, 0.8};
  std::uniform_real_distribution<> higher_coef_dis{-0.01, 0.01};
  const double a0{(*min_radius - eps_dis(*generator) * *max_radius) /
                  (0.25 * M_2_SQRTPI)};
  const std::array<DataVector, 4> initial_coefficients{
      {{{a0}},
       {{higher_coef_dis(*generator) * a0}},
       {{higher_coef_dis(*generator) * a0}},
       {{higher_coef_dis(*generator) * a0}}}};
  std::uniform_real_distribution<> dt_dis{0.1, 0.5};
  const double initial_time{*time - dt_dis(*generator)};
  const double expiration_time{*time + dt_dis(*generator)};
  (*functions_of_time)[f_of_t_name] =
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<3>>(
          initial_time, initial_coefficients, expiration_time);
}

// Generates the map, time, an a FunctionOfTime, but hides internal details (min
// radius, max radius, and center) when not needed
void generate_map_time_and_f_of_time(
    gsl::not_null<
        domain::CoordinateMaps::TimeDependent::SphericalCompression<3>*>
        map,
    gsl::not_null<double*> time,
    gsl::not_null<std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>*>
        functions_of_time,
    gsl::not_null<std::mt19937*> generator) noexcept {
  double min_radius{std::numeric_limits<double>::signaling_NaN()};
  double max_radius{std::numeric_limits<double>::signaling_NaN()};
  std::array<double, 3> center{};
  generate_map_time_and_f_of_time(
      map, time, functions_of_time, make_not_null(&min_radius),
      make_not_null(&max_radius), make_not_null(&center), generator);
}
}  // namespace

namespace domain {
namespace {
void test_suite(gsl::not_null<std::mt19937*> generator) noexcept {
  INFO("Suite");
  CoordinateMaps::TimeDependent::SphericalCompression<3> map{};
  double time{std::numeric_limits<double>::signaling_NaN()};
  std::unordered_map<std::string,
                     std::unique_ptr<::domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};

  generate_map_time_and_f_of_time(make_not_null(&map), make_not_null(&time),
                                  make_not_null(&functions_of_time), generator);
  test_suite_for_time_dependent_map_on_sphere(map, time, functions_of_time,
                                              generator);
}

void test_map(gsl::not_null<std::mt19937*> generator) noexcept {
  INFO("Map");
  std::uniform_real_distribution<> radius_dis(0, 1);
  std::uniform_real_distribution<> phi_dis(0, 2.0 * M_PI);
  std::uniform_real_distribution<> theta_dis(0, M_PI);

  // Create a map, choose a time, and create a FunctionOfTime.
  // Set up the map
  CoordinateMaps::TimeDependent::SphericalCompression<3> map{};
  double time{std::numeric_limits<double>::signaling_NaN()};
  std::unordered_map<std::string,
                     std::unique_ptr<::domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};

  double min_radius{std::numeric_limits<double>::signaling_NaN()};
  double max_radius{std::numeric_limits<double>::signaling_NaN()};
  std::array<double, 3> center{std::numeric_limits<double>::signaling_NaN()};
  generate_map_time_and_f_of_time(
      make_not_null(&map), make_not_null(&time),
      make_not_null(&functions_of_time), make_not_null(&min_radius),
      make_not_null(&max_radius), make_not_null(&center), generator);

  const double theta{theta_dis(*generator)};
  const double phi{phi_dis(*generator)};

  // Chose a point of unit radius with the randomly selected theta, phi.
  // This test will rescale this point to place it in different regions.
  const double rho_x0{sin(theta) * cos(phi)};
  const double rho_y0{sin(theta) * sin(phi)};
  const double rho_z0{cos(theta)};

  // A helper that returns a point given a radius
  auto point = [&rho_x0, &rho_y0, &rho_z0, &center](const double& radius) {
    return std::array<double, 3>{{radius * rho_x0 + center[0],
                                  radius * rho_y0 + center[1],
                                  radius * rho_z0 + center[2]}};
  };

  // A helper that checks if two points are scaled by the same factor
  // or not
  auto check_point_scale_factors =
      [&center, &map, &time, &functions_of_time](
          const std::array<double, 3>& orig_point_1,
          const std::array<double, 3>& orig_point_2,
          const bool check_if_equal) {
        const std::array<double, 3> mapped_point_1{
            map(orig_point_1, time, functions_of_time)};
        const std::array<double, 3> mapped_point_2{
            map(orig_point_2, time, functions_of_time)};
        const std::array<double, 3> scale_factor_1{
            {(orig_point_1[0] - mapped_point_1[0]) /
                 (orig_point_1[0] - center[0]),
             (orig_point_1[1] - mapped_point_1[1]) /
                 (orig_point_1[1] - center[1]),
             (orig_point_1[2] - mapped_point_1[2]) /
                 (orig_point_1[2] - center[2])}};
        const std::array<double, 3> scale_factor_2{
            {(orig_point_2[0] - mapped_point_2[0]) /
                 (orig_point_2[0] - center[0]),
             (orig_point_2[1] - mapped_point_2[1]) /
                 (orig_point_2[1] - center[1]),
             (orig_point_2[2] - mapped_point_2[2]) /
                 (orig_point_2[2] - center[2])}};
        Approx custom_approx = Approx::custom().epsilon(1.0e-9).scale(1.0);
        if (check_if_equal) {
          CHECK(scale_factor_1[0] == custom_approx(scale_factor_2[0]));
          CHECK(scale_factor_1[1] == custom_approx(scale_factor_2[1]));
          CHECK(scale_factor_1[2] == custom_approx(scale_factor_2[2]));
        } else {
          CHECK(scale_factor_1[0] != custom_approx(scale_factor_2[0]));
          CHECK(scale_factor_1[1] != custom_approx(scale_factor_2[1]));
          CHECK(scale_factor_1[2] != custom_approx(scale_factor_2[2]));
        }
      };

  // Set up distributions for choosing radii in the three regions:
  // smaller than min_radius, larger than max_radius, and in between
  constexpr double eps{0.01};
  std::uniform_real_distribution<> interior_dis{eps, min_radius - eps};
  std::uniform_real_distribution<> middle_dis{min_radius + eps,
                                              max_radius - eps};
  std::uniform_real_distribution<> exterior_dis{max_radius + eps,
                                                2.0 * max_radius};

  // In the interior, two random points should change radius by the same
  // factor, and that factor is the same as for a point exactly at
  // the minimum radius
  check_point_scale_factors(point(interior_dis(*generator)),
                            point(interior_dis(*generator)), true);
  check_point_scale_factors(point(interior_dis(*generator)), point(min_radius),
                            true);

  // In the middle, two points should only move radially, but by
  // a different amount
  check_point_scale_factors(point(middle_dis(*generator)),
                            point(middle_dis(*generator)), false);

  // In exterior and at max_radius, the map should be the identity
  const std::array<double, 3> exterior_point{point(exterior_dis(*generator))};
  const std::array<double, 3> max_rad_point{point(max_radius)};
  CHECK_ITERABLE_APPROX(exterior_point,
                        map(exterior_point, time, functions_of_time));
  CHECK_ITERABLE_APPROX(max_rad_point,
                        map(max_rad_point, time, functions_of_time));

  // Check that a point is invertible with the supplied functions_of_time
  CHECK(static_cast<bool>(
      map.inverse(point(middle_dis(*generator)), time, functions_of_time)));

  // Check that a point is not invertible if the function of time is
  // too big or too small
  std::uniform_real_distribution<> higher_coef_dis{-0.01, 0.01};
  const double a0{(min_radius - max_radius - 0.5) / (0.25 * M_2_SQRTPI)};
  const std::array<DataVector, 4> initial_coefficients{
      {{{a0}},
       {{higher_coef_dis(*generator) * a0}},
       {{higher_coef_dis(*generator) * a0}},
       {{higher_coef_dis(*generator) * a0}}}};
  const double b0{(min_radius + 0.5) / (0.25 * M_2_SQRTPI)};
  const std::array<DataVector, 4> initial_coefficients_b{
      {{{b0}},
       {{higher_coef_dis(*generator) * b0}},
       {{higher_coef_dis(*generator) * b0}},
       {{higher_coef_dis(*generator) * b0}}}};
  std::uniform_real_distribution<> dt_dis{0.1, 0.5};
  const double initial_time{time - dt_dis(*generator)};
  const double expiration_time{time + dt_dis(*generator)};

  const std::string f_of_t_name{"ExpansionFactor"};
  functions_of_time[f_of_t_name] =
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<3>>(
          initial_time, initial_coefficients, expiration_time);
  CHECK_FALSE(static_cast<bool>(
      map.inverse(point(middle_dis(*generator)), time, functions_of_time)));
  functions_of_time[f_of_t_name] =
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<3>>(
          initial_time, initial_coefficients_b, expiration_time);
  CHECK_FALSE(static_cast<bool>(
      map.inverse(point(middle_dis(*generator)), time, functions_of_time)));
}

void test_is_identity(gsl::not_null<std::mt19937*> generator) noexcept {
  INFO("Is identity");
  CoordinateMaps::TimeDependent::SphericalCompression<3> map{};
  double time{std::numeric_limits<double>::signaling_NaN()};
  std::unordered_map<std::string,
                     std::unique_ptr<::domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};
  generate_map_time_and_f_of_time(make_not_null(&map), make_not_null(&time),
                                  make_not_null(&functions_of_time), generator);
  CHECK(not map.is_identity());
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.SphericalCompression",
                  "[Domain][Unit]") {
  MAKE_GENERATOR(gen, 1989166745);
  test_suite(make_not_null(&gen));
  test_map(make_not_null(&gen));
  test_is_identity(make_not_null(&gen));
}
}  // namespace domain
