// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <string>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/RotatingQuadrupole.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"

namespace {
using gh::BoundaryConditions::RotatingQuadrupole;

void test_envelope_and_carrier() {
  const auto wave = TestHelpers::test_creation<RotatingQuadrupole>(
      "Amplitude: 0.2\n"
      "AngularFrequency: 0.9\n"
      "Phase: 0.3\n"
      "StartTime: 10.0\n"
      "RampUpDuration: 4.0\n"
      "StopTime: 30.0\n"
      "RampDownDuration: 6.0\n");
  const auto restored = serialize_and_deserialize(wave);
  const tnsr::aa<double, 3> zero{0.};
  for (const double time : {-100., 10., 36., 100.}) {
    CHECK(wave.envelope(time) == 0.);
    CHECK(wave(time) == zero);
  }
  CHECK(wave.envelope(12.) == approx(0.5));
  CHECK(wave.envelope(33.) == approx(0.5));
  for (const double time : {14., 19., 30.}) {
    CHECK(wave.envelope(time) == 1.);
  }
  // Both derivatives vanish at every ramp join. Near each join the departure
  // from its limiting value scales as the cube of the time separation.
  const double epsilon = 1.e-3;
  CHECK(wave.envelope(10. + epsilon) < 2.e-10);
  CHECK(1. - wave.envelope(14. - epsilon) < 2.e-10);
  CHECK(1. - wave.envelope(30. + epsilon) < 5.e-11);
  CHECK(wave.envelope(36. - epsilon) < 5.e-11);
  for (const double time : {9., 10.5, 13., 14., 23., 30.1, 35., 36.}) {
    CHECK_ITERABLE_APPROX(restored(time), wave(time));
    const auto tensor = wave(time);
    CHECK(get<1, 1>(tensor) ==
          approx(0.2 * wave.envelope(time) * cos(0.9 * (time - 10.) + 0.3)));
    CHECK(get<1, 2>(tensor) ==
          approx(0.2 * wave.envelope(time) * sin(0.9 * (time - 10.) + 0.3)));
    CHECK(get<2, 2>(tensor) == -get<1, 1>(tensor));
    CHECK(get<0, 0>(tensor) == 0.);
    CHECK(get<0, 1>(tensor) == 0.);
    CHECK(get<0, 2>(tensor) == 0.);
    CHECK(get<0, 3>(tensor) == 0.);
    CHECK(get<1, 3>(tensor) == 0.);
    CHECK(get<2, 3>(tensor) == 0.);
    CHECK(get<3, 3>(tensor) == 0.);
  }
  const double period = 2. * M_PI / 0.9;
  CHECK_ITERABLE_APPROX(wave(15. + period), wave(15.));
  const auto forever = TestHelpers::test_creation<RotatingQuadrupole>(
      "Amplitude: 0.2\n"
      "AngularFrequency: -0.9\n"
      "Phase: 0.0\n"
      "StartTime: 10.0\n"
      "RampUpDuration: 4.0\n"
      "StopTime: None\n"
      "RampDownDuration: 6.0\n");
  CHECK(forever.envelope(1000.) == 1.);
  CHECK_ITERABLE_APPROX(serialize_and_deserialize(forever)(1000.),
                        forever(1000.));
  const RotatingQuadrupole disabled{0., 0.9, 0., 0., 1., std::nullopt, 1.};
  CHECK(disabled(3.) == zero);
}

// Contract with the orthonormal spherical polarizations. This independently
// checks the angular m=2 dependence and sign of the rotating pattern; a scalar
// sinusoid multiplying the legacy axisymmetric tensor fails this test.
void test_chirality() {
  const double amplitude = 0.07;
  const double frequency = 0.8;
  const double phase = 0.3;
  const RotatingQuadrupole wave{amplitude, frequency,    phase, 0.,
                                1.,        std::nullopt, 1.};
  const RotatingQuadrupole reverse{amplitude, -frequency,   -phase, 0.,
                                   1.,        std::nullopt, 1.};
  for (const double time : {2., 3.5, 7.}) {
    const auto q = wave(time);
    CHECK(get<1, 1>(reverse(time)) == approx(get<1, 1>(q)));
    CHECK(get<1, 2>(reverse(time)) == approx(-get<1, 2>(q)));
    for (const double theta : {0.2, 0.7, 1.5, 2.8}) {
      for (const double phi : {0., M_PI / 4., 1.2}) {
        const std::array<double, 3> e_theta{cos(theta) * cos(phi),
                                            cos(theta) * sin(phi), -sin(theta)};
        const std::array<double, 3> e_phi{-sin(phi), cos(phi), 0.};
        double plus = 0.;
        double cross = 0.;
        for (size_t i = 0; i < 3; ++i) {
          for (size_t j = 0; j < 3; ++j) {
            plus += 0.5 *
                    (gsl::at(e_theta, i) * gsl::at(e_theta, j) -
                     gsl::at(e_phi, i) * gsl::at(e_phi, j)) *
                    q.get(i + 1, j + 1);
            cross +=
                gsl::at(e_theta, i) * gsl::at(e_phi, j) * q.get(i + 1, j + 1);
          }
        }
        const double angular_phase = 2. * phi - frequency * time - phase;
        CHECK(plus == approx(0.5 * amplitude * (1. + square(cos(theta))) *
                             cos(angular_phase)));
        CHECK(cross == approx(-amplitude * cos(theta) * sin(angular_phase)));
      }
    }
  }
}

void test_invalid_options() {
  const std::string message = "RotatingQuadrupole requires finite parameters";
  for (const double amplitude : {-1., std::numeric_limits<double>::infinity(),
                                 std::numeric_limits<double>::quiet_NaN()}) {
    CHECK_THROWS_WITH(
        (RotatingQuadrupole{amplitude, 1., 0., 0., 1., std::nullopt, 1.}),
        Catch::Matchers::ContainsSubstring(message));
  }
  for (const double frequency : {0., std::numeric_limits<double>::infinity(),
                                 std::numeric_limits<double>::quiet_NaN()}) {
    CHECK_THROWS_WITH(
        (RotatingQuadrupole{1., frequency, 0., 0., 1., std::nullopt, 1.}),
        Catch::Matchers::ContainsSubstring(message));
  }
  for (const double duration :
       {0., -1., std::numeric_limits<double>::infinity()}) {
    CHECK_THROWS_WITH(
        (RotatingQuadrupole{1., 1., 0., 0., duration, std::nullopt, 1.}),
        Catch::Matchers::ContainsSubstring(message));
    CHECK_THROWS_WITH(
        (RotatingQuadrupole{1., 1., 0., 0., 1., std::nullopt, duration}),
        Catch::Matchers::ContainsSubstring(message));
  }
  for (const double stop : {0.5, std::numeric_limits<double>::infinity(),
                            std::numeric_limits<double>::quiet_NaN()}) {
    CHECK_THROWS_WITH((RotatingQuadrupole{1., 1., 0., 0., 1., stop, 1.}),
                      Catch::Matchers::ContainsSubstring(
                          "RotatingQuadrupole StopTime must be finite"));
  }
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.GeneralizedHarmonic.RotatingQuadrupole",
    "[Unit][Evolution]") {
  test_envelope_and_carrier();
  test_chirality();
  test_invalid_options();
}
