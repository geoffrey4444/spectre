// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <deque>
#include <optional>
#include <unordered_set>
#include <utility>

#include "DataStructures/LinkedMessageId.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackIterator.hpp"
#include "NumericalAlgorithms/Strahlkorper/Strahlkorper.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Storage.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/TimeExtrapolation.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"

namespace ah {
namespace {
using Surface = ylm::Strahlkorper<Frame::Distorted>;
using History = std::deque<Storage::PreviousSurface<Frame::Distorted>>;

Surface make_surface(const size_t l_max, const double quadrupole) {
  Surface surface{l_max, 2.0, {0.0, 0.0, 0.0}};
  ylm::SpherepackIterator mode(l_max, l_max);
  surface.coefficients()[mode.set(2, -1)()] = quadrupole;
  return surface;
}

void test_cubic_and_startup() {
  // The newly available point count first needs to predict a known surface
  // before it can be selected. Use nonuniform times and changing resolutions.
  const std::array times{0., 0.1, 0.4, 0.6, 0.8, 1.2};
  const std::array<size_t, 6> resolutions{2, 3, 4, 3, 5, 2};
  const std::array<size_t, 6> expected_points{0, 1, 2, 2, 3, 4};
  History history{};
  PredictionErrors errors{};
  for (size_t i = 0; i < times.size(); ++i) {
    CAPTURE(i);
    const double time = gsl::at(times, i);
    const size_t points = number_of_points_for_extrapolation(
        std::nullopt, history.size(), errors);
    CHECK(points == gsl::at(expected_points, i));
    auto found = make_surface(gsl::at(resolutions, i), 0.1 * cube(time));
    if (points == 4) {
      CHECK_ITERABLE_APPROX(extrapolate_surface_coefficients(
                                time, points, history, found.ylm_spherepack()),
                            found.coefficients());
    }
    errors = prediction_errors(time, found, history);
    for (size_t n = 2; n <= 4; ++n) {
      CHECK(gsl::at(errors, n - 2).has_value() == (n <= history.size()));
    }
    history.emplace_front(
        LinkedMessageId<double>{.id = time, .previous = std::nullopt},
        std::move(found), std::unordered_set<ElementId<3>>{});
    if (history.size() > 4) {
      history.pop_back();
    }
  }
  // A new high mode from an increase in L is outside the common comparison
  // basis of every prediction. Only the predictable cubic mode contributes.
  auto found = make_surface(5, 0.1 * cube(1.5));
  ylm::SpherepackIterator mode(5, 5);
  found.coefficients()[mode.set(4, 0)()] = 1.0;
  errors = prediction_errors(1.5, found, history);
  REQUIRE(errors[2].has_value());
  CHECK(*errors[2] == approx(0.0));
  CHECK(number_of_points_for_extrapolation(std::nullopt, 4, errors) == 4);
}

void test_noise() {
  // At equal intervals, linear, quadratic and cubic extrapolation of an
  // alternating mode miss the next sample by 4, 8 and 16 times its amplitude.
  // Only one of the 16 real coefficients at L=3 contributes to the RMS.
  History history{};
  const double noise = 1.e-5;
  for (size_t i = 0; i < 4; ++i) {
    history.emplace_front(LinkedMessageId<double>{.id = static_cast<double>(i),
                                                  .previous = std::nullopt},
                          make_surface(3, i % 2 == 0 ? noise : -noise),
                          std::unordered_set<ElementId<3>>{});
  }
  auto found = make_surface(3, noise);
  // Changing unused SPHEREPACK padding cannot contribute to the score.
  const size_t padding = found.coefficients().size() / 2;
  REQUIRE_FALSE(ylm::SpherepackIterator(3, 3).compact_index(padding));
  found.coefficients()[padding] = 10.0;
  const auto errors = prediction_errors(4., found, history);
  REQUIRE(errors[0].has_value());
  REQUIRE(errors[1].has_value());
  REQUIRE(errors[2].has_value());
  CHECK(*errors[0] == approx(noise));
  CHECK(*errors[1] == approx(2. * noise));
  CHECK(*errors[2] == approx(4. * noise));
  CHECK(number_of_points_for_extrapolation(std::nullopt, 4, errors) == 2);
}

void test_mixed_resolutions() {
  History history{};
  auto newest = make_surface(3, 0.);
  ylm::SpherepackIterator mode(3, 3);
  newest.coefficients()[mode.set(3, -2)()] = 0.01;
  history.emplace_back(
      LinkedMessageId<double>{.id = 1.0, .previous = std::nullopt},
      std::move(newest), std::unordered_set<ElementId<3>>{});
  history.emplace_back(
      LinkedMessageId<double>{.id = 0.0, .previous = std::nullopt},
      make_surface(2, 0.), std::unordered_set<ElementId<3>>{});
  // The mode absent from the older surface contributes zero at that time.
  auto expected = make_surface(4, 0.);
  ylm::SpherepackIterator target_mode(4, 4);
  expected.coefficients()[target_mode.set(3, -2)()] = 0.02;
  CHECK_ITERABLE_APPROX(extrapolate_surface_coefficients(
                            2., 2, history, expected.ylm_spherepack()),
                        expected.coefficients());
}
}  // namespace

SPECTRE_TEST_CASE("Unit.ApparentHorizonFinder.TimeExtrapolation",
                  "[ApparentHorizonFinder][Unit]") {
  test_cubic_and_startup();
  test_noise();
  test_mixed_resolutions();
  const PredictionErrors tied{2., 1., 1.};
  CHECK(number_of_points_for_extrapolation(std::nullopt, 4, tied) == 3);
  CHECK(number_of_points_for_extrapolation(std::nullopt, 2, tied) == 2);
  CHECK(number_of_points_for_extrapolation(std::nullopt, 4, {}) == 4);
  CHECK(number_of_points_for_extrapolation(4, 4, tied) == 4);
  CHECK(number_of_points_for_extrapolation(3, 4, tied) == 3);
  CHECK(number_of_points_for_extrapolation(2, 4, tied) == 2);
  CHECK(number_of_points_for_extrapolation(3, 1, tied) == 1);
}
}  // namespace ah
