// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <limits>
#include <string>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/AnalyticSolutions/TestHelpers.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/TeukolskyWave.hpp"
#include "PointwiseFunctions/GeneralRelativity/ExtrinsicCurvature.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace {

template <typename DataType>
void test_teukolsky_wave(const gr::Solutions::TeukolskyWave& solution,
                         const DataType& used_for_size) {
  auto x = make_with_value<tnsr::I<DataType, 3, Frame::Inertial>>(used_for_size,
                                                                  0.0);
  get<0>(x) = make_with_value<DataType>(used_for_size, 6.2);
  get<1>(x) = make_with_value<DataType>(used_for_size, -1.4);
  get<2>(x) = make_with_value<DataType>(used_for_size, 3.6);
  const double t = 1.3;

  const auto vars = solution.variables(
      x, t, typename gr::Solutions::TeukolskyWave::template tags<DataType>{});
  const auto& lapse = get<gr::Tags::Lapse<DataType>>(vars);
  const auto& dt_lapse = get<Tags::dt<gr::Tags::Lapse<DataType>>>(vars);
  const auto& deriv_lapse =
      get<::Tags::deriv<gr::Tags::Lapse<DataType>, tmpl::size_t<3>,
                        Frame::Inertial>>(vars);
  const auto& shift = get<gr::Tags::Shift<DataType, 3>>(vars);
  const auto& dt_shift = get<Tags::dt<gr::Tags::Shift<DataType, 3>>>(vars);
  const auto& deriv_shift =
      get<::Tags::deriv<gr::Tags::Shift<DataType, 3>, tmpl::size_t<3>,
                        Frame::Inertial>>(vars);
  const auto& spatial_metric = get<gr::Tags::SpatialMetric<DataType, 3>>(vars);
  const auto& dt_spatial_metric =
      get<Tags::dt<gr::Tags::SpatialMetric<DataType, 3>>>(vars);
  const auto& deriv_spatial_metric =
      get<::Tags::deriv<gr::Tags::SpatialMetric<DataType, 3>, tmpl::size_t<3>,
                        Frame::Inertial>>(vars);
  const auto zero_deriv_lapse =
      make_with_value<tnsr::i<DataType, 3, Frame::Inertial>>(x, 0.0);
  const auto zero_shift =
      make_with_value<tnsr::I<DataType, 3, Frame::Inertial>>(x, 0.0);
  const auto zero_deriv_shift =
      make_with_value<tnsr::iJ<DataType, 3, Frame::Inertial>>(x, 0.0);

  CHECK_ITERABLE_APPROX(get(lapse),
                        make_with_value<DataType>(used_for_size, 1.0));
  CHECK_ITERABLE_APPROX(get(dt_lapse),
                        make_with_value<DataType>(used_for_size, 0.0));
  CHECK_ITERABLE_APPROX(deriv_lapse, zero_deriv_lapse);
  CHECK_ITERABLE_APPROX(shift, zero_shift);
  CHECK_ITERABLE_APPROX(dt_shift, zero_shift);
  CHECK_ITERABLE_APPROX(deriv_shift, zero_deriv_shift);

  const auto det_and_inverse = determinant_and_inverse(spatial_metric);
  const auto& inverse_spatial_metric =
      get<gr::Tags::InverseSpatialMetric<DataType, 3>>(vars);
  CHECK_ITERABLE_APPROX(inverse_spatial_metric, det_and_inverse.second);
  CHECK_ITERABLE_APPROX(
      get(get<gr::Tags::SqrtDetSpatialMetric<DataType>>(vars)),
      sqrt(get(det_and_inverse.first)));

  const auto expected_extrinsic_curvature =
      gr::extrinsic_curvature(lapse, shift, deriv_shift, spatial_metric,
                              dt_spatial_metric, deriv_spatial_metric);
  const auto& extrinsic_curvature =
      get<gr::Tags::ExtrinsicCurvature<DataType, 3>>(vars);
  CHECK_ITERABLE_APPROX(extrinsic_curvature, expected_extrinsic_curvature);

  tmpl::for_each<
      typename gr::Solutions::TeukolskyWave::template tags<DataType>>(
      [&solution, &vars, &x, &t](auto tag_v) {
        using Tag = tmpl::type_from<decltype(tag_v)>;
        CHECK_ITERABLE_APPROX(get<Tag>(vars), get<Tag>(solution.variables(
                                                  x, t, tmpl::list<Tag>{})));
      });

  TestHelpers::AnalyticSolutions::test_tag_retrieval(
      solution, x, t,
      typename gr::Solutions::TeukolskyWave::template tags<DataType>{});
}

void test_serialize() {
  const gr::Solutions::TeukolskyWave solution(1.0e-4, 2, "even", "outgoing",
                                              {{0.0, 0.0, 0.0}}, 8.0, 1.5);
  test_serialization(solution);
  test_teukolsky_wave(serialize_and_deserialize(solution), DataVector{5});
}

void test_copy_and_move() {
  gr::Solutions::TeukolskyWave solution(1.0e-4, 2, "even", "outgoing",
                                        {{0.0, 0.0, 0.0}}, 8.0, 1.5);
  test_copy_semantics(solution);
  auto solution_copy = solution;
  test_move_semantics(std::move(solution), solution_copy);  // NOLINT
}

void test_construct_from_options() {
  const auto created = TestHelpers::test_creation<gr::Solutions::TeukolskyWave>(
      "Amplitude: 1e-4\n"
      "Mode: 2\n"
      "Parity: even\n"
      "Direction: outgoing\n"
      "Center: [0.0, 0.0, 0.0]\n"
      "Radius: 8.0\n"
      "Width: 1.5");
  CHECK(created == gr::Solutions::TeukolskyWave(1.0e-4, 2, "even", "outgoing",
                                                {{0.0, 0.0, 0.0}}, 8.0, 1.5));
}

}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticSolutions.Gr.TeukolskyWave",
                  "[PointwiseFunctions][Unit]") {
  const gr::Solutions::TeukolskyWave solution(1.0e-4, 2, "even", "outgoing",
                                              {{0.0, 0.0, 0.0}}, 8.0, 1.5);
  test_teukolsky_wave(
      solution, DataVector(5, std::numeric_limits<double>::signaling_NaN()));
  test_teukolsky_wave(solution, std::numeric_limits<double>::signaling_NaN());
  test_serialize();
  test_copy_and_move();
  test_construct_from_options();

  CHECK_THROWS_WITH(
      []() {
        const gr::Solutions::TeukolskyWave bad_solution(
            1.0e-4, 3, "even", "outgoing", {{0.0, 0.0, 0.0}}, 8.0, 1.5);
      }(),
      Catch::Matchers::ContainsSubstring("Mode must lie between -2 and 2"));
}
