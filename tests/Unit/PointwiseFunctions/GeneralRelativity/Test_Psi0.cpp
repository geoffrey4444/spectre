// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <random>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/PointwiseFunctions/GeneralRelativity/TestHelpers.hpp"
#include "PointwiseFunctions/GeneralRelativity/Psi0Real.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"

namespace {
void test_psi_0_real() {
  TestHelpers::db::test_compute_tag<gr::Tags::Psi0RealCompute<Frame::Inertial>>(
      "Psi0Real");

  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> distribution(0.1, 3.0);
  const auto nn_generator = make_not_null(&generator);
  const auto nn_distribution = make_not_null(&distribution);
  const DataVector used_for_size(5,
                                 std::numeric_limits<double>::signaling_NaN());

  const auto spatial_ricci =
      make_with_random_values<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          nn_generator, nn_distribution, used_for_size);
  const auto extrinsic_curvature =
      make_with_random_values<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          nn_generator, nn_distribution, used_for_size);
  const auto cov_deriv_extrinsic_curvature =
      make_with_random_values<tnsr::ijj<DataVector, 3, Frame::Inertial>>(
          nn_generator, nn_distribution, used_for_size);
  const auto spatial_metric =
      TestHelpers::gr::random_spatial_metric<3, DataVector, Frame::Inertial>(
          nn_generator, used_for_size);
  const auto inverse_spatial_metric =
      determinant_and_inverse(spatial_metric).second;
  const auto inertial_coords =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
          nn_generator, nn_distribution, used_for_size);

  const auto expected = pypp::call<Scalar<DataVector>>(
      "GeneralRelativity.Psi0", "psi_0_real", spatial_ricci,
      extrinsic_curvature, cov_deriv_extrinsic_curvature, spatial_metric,
      inverse_spatial_metric, inertial_coords);
  CHECK_ITERABLE_APPROX(
      gr::psi_0_real(spatial_ricci, extrinsic_curvature,
                     cov_deriv_extrinsic_curvature, spatial_metric,
                     inverse_spatial_metric, inertial_coords),
      expected);

  auto result = make_with_value<Scalar<DataVector>>(
      used_for_size, std::numeric_limits<double>::signaling_NaN());
  gr::psi_0_real(make_not_null(&result), spatial_ricci, extrinsic_curvature,
                 cov_deriv_extrinsic_curvature, spatial_metric,
                 inverse_spatial_metric, inertial_coords);
  CHECK_ITERABLE_APPROX(result, expected);

  const auto box = db::create<
      db::AddSimpleTags<
          gr::Tags::SpatialRicci<DataVector, 3>,
          gr::Tags::ExtrinsicCurvature<DataVector, 3>,
          ::Tags::deriv<gr::Tags::ExtrinsicCurvature<DataVector, 3>,
                        tmpl::size_t<3>, Frame::Inertial>,
          gr::Tags::SpatialMetric<DataVector, 3>,
          gr::Tags::InverseSpatialMetric<DataVector, 3>,
          domain::Tags::Coordinates<3, Frame::Inertial>>,
      db::AddComputeTags<gr::Tags::Psi0RealCompute<Frame::Inertial>>>(
      spatial_ricci, extrinsic_curvature, cov_deriv_extrinsic_curvature,
      spatial_metric, inverse_spatial_metric, inertial_coords);
  CHECK_ITERABLE_APPROX((db::get<gr::Tags::Psi0Real<DataVector>>(box)),
                        expected);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.GeneralRelativity.Psi0Real",
                  "[Unit][PointwiseFunctions]") {
  pypp::SetupLocalPythonEnvironment local_python_env("PointwiseFunctions/");
  test_psi_0_real();
}
