// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
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
#include "PointwiseFunctions/GeneralRelativity/ExtrinsicCurvature.hpp"
#include "PointwiseFunctions/GeneralRelativity/Psi0Real.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "Utilities/MakeWithValue.hpp"

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

  const auto christoffel =
      make_with_random_values<tnsr::Ijj<DataVector, 3, Frame::Inertial>>(
          nn_generator, nn_distribution, used_for_size);
  auto partial_deriv_extrinsic_curvature = cov_deriv_extrinsic_curvature;
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      for (size_t k = j; k < 3; ++k) {
        for (size_t l = 0; l < 3; ++l) {
          partial_deriv_extrinsic_curvature.get(i, j, k) +=
              christoffel.get(l, i, j) * extrinsic_curvature.get(l, k) +
              christoffel.get(l, i, k) * extrinsic_curvature.get(l, j);
        }
      }
    }
  }
  // Distinguish partial and covariant derivatives in the observation path.
  const auto box =
      db::create<db::AddSimpleTags<
                     gr::Tags::SpatialRicci<DataVector, 3>,
                     gr::Tags::ExtrinsicCurvature<DataVector, 3>,
                     ::Tags::deriv<gr::Tags::ExtrinsicCurvature<DataVector, 3>,
                                   tmpl::size_t<3>, Frame::Inertial>,
                     gr::Tags::SpatialChristoffelSecondKind<DataVector, 3>,
                     gr::Tags::SpatialMetric<DataVector, 3>,
                     gr::Tags::InverseSpatialMetric<DataVector, 3>,
                     domain::Tags::Coordinates<3, Frame::Inertial>>,
                 db::AddComputeTags<
                     gr::Tags::CovariantDerivativeOfExtrinsicCurvatureCompute<
                         3, Frame::Inertial>,
                     gr::Tags::Psi0RealCompute<Frame::Inertial>>>(
          spatial_ricci, extrinsic_curvature, partial_deriv_extrinsic_curvature,
          christoffel, spatial_metric, inverse_spatial_metric, inertial_coords);
  CHECK_ITERABLE_APPROX((db::get<gr::Tags::Psi0Real<DataVector>>(box)),
                        expected);
}

void test_polarization_basis_on_coordinate_planes() {
  const DataVector used_for_size(7, 0.0);
  auto spatial_ricci =
      make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(used_for_size,
                                                                0.0);
  const auto extrinsic_curvature =
      make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(used_for_size,
                                                                0.0);
  const auto cov_deriv_extrinsic_curvature =
      make_with_value<tnsr::ijj<DataVector, 3, Frame::Inertial>>(used_for_size,
                                                                 0.0);
  auto spatial_metric =
      make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(used_for_size,
                                                                0.0);
  auto inverse_spatial_metric =
      make_with_value<tnsr::II<DataVector, 3, Frame::Inertial>>(used_for_size,
                                                                0.0);
  for (size_t i = 0; i < 3; ++i) {
    spatial_metric.get(i, i) = 1.0;
    inverse_spatial_metric.get(i, i) = 1.0;
  }
  // At the first four points the radial direction is in the xy plane. The
  // projected x direction and its metric cross product with the radial
  // direction form an orthonormal polarization basis. For this Ricci tensor,
  // the transverse trace-free projection therefore gives Re(Psi0) = 1/2 at
  // every azimuth, including on the x axis where the projected y direction is
  // used as a fallback.
  get<2, 2>(spatial_ricci) = DataVector{1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0};
  // The next two points check the historical Cartesian polarization basis on
  // the positive and negative z axes. The final point checks that basis at the
  // origin. At all three points Re(Psi0) = -1/2 for Ricci_xx = 1.
  get<0, 0>(spatial_ricci) = DataVector{0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0};
  auto inertial_coords =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(used_for_size,
                                                               0.0);
  const double inverse_sqrt_two = 1.0 / sqrt(2.0);
  get<0>(inertial_coords) =
      DataVector{1.0, inverse_sqrt_two, 0.0, -inverse_sqrt_two, 0.0, 0.0, 0.0};
  get<1>(inertial_coords) =
      DataVector{0.0, inverse_sqrt_two, 1.0, inverse_sqrt_two, 0.0, 0.0, 0.0};
  get<2>(inertial_coords) = DataVector{0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0};

  const auto result = gr::psi_0_real(
      spatial_ricci, extrinsic_curvature, cov_deriv_extrinsic_curvature,
      spatial_metric, inverse_spatial_metric, inertial_coords);
  auto expected = DataVector(7, 0.5);
  expected[4] = expected[5] = expected[6] = -0.5;
  CHECK_ITERABLE_APPROX(get(result), expected);
}

void test_polarization_basis_with_curved_metric() {
  const DataVector used_for_size(1, 0.0);
  auto spatial_ricci =
      make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(used_for_size,
                                                                0.0);
  const auto extrinsic_curvature =
      make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(used_for_size,
                                                                0.0);
  const auto cov_deriv_extrinsic_curvature =
      make_with_value<tnsr::ijj<DataVector, 3, Frame::Inertial>>(used_for_size,
                                                                 0.0);
  auto spatial_metric =
      make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(used_for_size,
                                                                0.0);
  get<0, 0>(spatial_metric) = 1.0;
  get<0, 1>(spatial_metric) = 0.5;
  get<1, 1>(spatial_metric) = 1.0;
  get<2, 2>(spatial_metric) = 4.0;
  auto inverse_spatial_metric =
      make_with_value<tnsr::II<DataVector, 3, Frame::Inertial>>(used_for_size,
                                                                0.0);
  get<0, 0>(inverse_spatial_metric) = 4.0 / 3.0;
  get<0, 1>(inverse_spatial_metric) = -2.0 / 3.0;
  get<1, 1>(inverse_spatial_metric) = 4.0 / 3.0;
  get<2, 2>(inverse_spatial_metric) = 0.25;
  auto inertial_coords =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(used_for_size,
                                                               0.0);
  get<0>(inertial_coords) = 1.0;

  // The projected x direction vanishes, so the first polarization is the
  // normalized projected y direction (-1/sqrt(3), 2/sqrt(3), 0). The curved
  // metric cross product must give the second polarization (0, 0, 1/2).
  get<2, 2>(spatial_ricci) = 4.0;
  const auto result = gr::psi_0_real(
      spatial_ricci, extrinsic_curvature, cov_deriv_extrinsic_curvature,
      spatial_metric, inverse_spatial_metric, inertial_coords);
  CHECK_ITERABLE_APPROX(get(result), DataVector(1, 0.5));
}

void test_nearly_degenerate_polarization_basis() {
  const DataVector used_for_size(6, 0.0);
  auto spatial_metric =
      make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(used_for_size,
                                                                0.0);
  auto inverse_spatial_metric =
      make_with_value<tnsr::II<DataVector, 3, Frame::Inertial>>(used_for_size,
                                                                0.0);
  for (size_t i = 0; i < 3; ++i) {
    spatial_metric.get(i, i) = 1.0;
    inverse_spatial_metric.get(i, i) = 1.0;
  }
  const auto extrinsic_curvature =
      make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(used_for_size,
                                                                0.0);
  const auto cov_deriv_extrinsic_curvature =
      make_with_value<tnsr::ijj<DataVector, 3, Frame::Inertial>>(used_for_size,
                                                                 0.0);
  auto spatial_ricci = spatial_metric;
  get<0, 0>(spatial_ricci) = 0.4;
  get<0, 1>(spatial_ricci) = 0.2;
  get<0, 2>(spatial_ricci) = 0.3;
  get<1, 1>(spatial_ricci) = -0.2;
  get<1, 2>(spatial_ricci) = 0.5;
  get<2, 2>(spatial_ricci) = 1.0;
  auto coords = make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
      used_for_size, 0.0);
  get<0>(coords) = 1.0;
  get<1>(coords) = DataVector{1.0, 1.0, 1.e-8, 1.e-8, 1.e-12, 1.e-12};
  get<2>(coords) = DataVector{1.e-12, -1.e-12, 1.e-8, -1.e-8, 1.e-12, -1.e-12};

  DataVector expected(used_for_size.size());
  for (size_t p = 0; p < used_for_size.size(); ++p) {
    const double y = get<1>(coords)[p];
    const double z = get<2>(coords)[p];
    const double transverse_radius = hypot(y, z);
    const double radius = hypot(1.0, transverse_radius);
    // The sign of the second polarization cancels in Re(Psi0).
    const double x_x = transverse_radius / radius;
    const double x_y = -y / (radius * transverse_radius);
    const double x_z = -z / (radius * transverse_radius);
    const double y_y = z / transverse_radius;
    const double y_z = -y / transverse_radius;
    expected[p] = -0.5 * (0.4 * x_x * x_x + 0.4 * x_x * x_y + 0.6 * x_x * x_z -
                          0.2 * (x_y * x_y - y_y * y_y) + x_y * x_z -
                          y_y * y_z + x_z * x_z - y_z * y_z);
  }
  CHECK_ITERABLE_APPROX(
      get(gr::psi_0_real(spatial_ricci, extrinsic_curvature,
                         cov_deriv_extrinsic_curvature, spatial_metric,
                         inverse_spatial_metric, coords)),
      expected);
}

void test_incoming_characteristic() {
  const DataVector used_for_size(2, 0.0);
  auto spatial_metric =
      make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(used_for_size,
                                                                0.0);
  auto inverse_spatial_metric =
      make_with_value<tnsr::II<DataVector, 3, Frame::Inertial>>(used_for_size,
                                                                0.0);
  for (size_t i = 0; i < 3; ++i) {
    spatial_metric.get(i, i) = 1.0;
    inverse_spatial_metric.get(i, i) = 1.0;
  }
  const auto extrinsic_curvature =
      make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(used_for_size,
                                                                0.0);
  auto cov_deriv_extrinsic_curvature =
      make_with_value<tnsr::ijj<DataVector, 3, Frame::Inertial>>(used_for_size,
                                                                 0.0);
  auto spatial_ricci =
      make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(used_for_size,
                                                                0.0);
  get<0, 0>(spatial_ricci) = 1.0;
  get<1, 1>(spatial_ricci) = -1.0;
  get<2, 0, 0>(cov_deriv_extrinsic_curvature) = DataVector{1.0, -1.0};
  get<2, 1, 1>(cov_deriv_extrinsic_curvature) = DataVector{-1.0, 1.0};
  auto coords = make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
      used_for_size, 0.0);
  get<2>(coords) = 1.0;
  // U8- adds the radial derivative of K_ij. The first point is a purely
  // incoming characteristic; the second has the same electric curvature
  // and the opposite derivative, so its incoming characteristic vanishes.
  CHECK_ITERABLE_APPROX(
      get(gr::psi_0_real(spatial_ricci, extrinsic_curvature,
                         cov_deriv_extrinsic_curvature, spatial_metric,
                         inverse_spatial_metric, coords)),
      (DataVector{-2.0, 0.0}));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.GeneralRelativity.Psi0Real",
                  "[Unit][PointwiseFunctions]") {
  const pypp::SetupLocalPythonEnvironment local_python_env(
      "PointwiseFunctions/");
  test_psi_0_real();
  test_polarization_basis_on_coordinate_planes();
  test_polarization_basis_with_curved_metric();
  test_nearly_degenerate_polarization_basis();
  test_incoming_characteristic();
}
