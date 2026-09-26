// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <random>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Expressions/Evaluate.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/PointwiseFunctions/GeneralRelativity/TestHelpers.hpp"
#include "PointwiseFunctions/GeneralRelativity/Psi4.hpp"
#include "PointwiseFunctions/GeneralRelativity/Psi4Imag.hpp"
#include "PointwiseFunctions/GeneralRelativity/Psi4Real.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "PointwiseFunctions/GeneralRelativity/WeylPropagating.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace {
template <typename RealDataType>
void test_compute_item_in_databox(const RealDataType& used_for_size_real) {
  TestHelpers::db::test_compute_tag<gr::Tags::Psi4RealCompute<Frame::Inertial>>(
      "Psi4Real");
  TestHelpers::db::test_compute_tag<gr::Tags::Psi4ImagCompute<Frame::Inertial>>(
      "Psi4Imag");

  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> distribution(0.1, 3.0);
  const auto nn_generator = make_not_null(&generator);
  const auto nn_distribution = make_not_null(&distribution);

  const auto spatial_ricci =
      make_with_random_values<tnsr::ii<RealDataType, 3, Frame::Inertial>>(
          nn_generator, nn_distribution, used_for_size_real);
  const auto extrinsic_curvature =
      make_with_random_values<tnsr::ii<RealDataType, 3, Frame::Inertial>>(
          nn_generator, nn_distribution, used_for_size_real);
  const auto cov_deriv_extrinsic_curvature =
      make_with_random_values<tnsr::ijj<RealDataType, 3, Frame::Inertial>>(
          nn_generator, nn_distribution, used_for_size_real);
  const auto spatial_metric =
      TestHelpers::gr::random_spatial_metric<3, RealDataType, Frame::Inertial>(
          nn_generator, used_for_size_real);
  const auto inv_spatial_metric =
      determinant_and_inverse(spatial_metric).second;
  const auto inertial_coords =
      make_with_random_values<tnsr::I<RealDataType, 3, Frame::Inertial>>(
          nn_generator, nn_distribution, used_for_size_real);

  const auto box = db::create<
      db::AddSimpleTags<
          gr::Tags::SpatialRicci<RealDataType, 3>,
          gr::Tags::ExtrinsicCurvature<RealDataType, 3>,
          ::Tags::deriv<gr::Tags::ExtrinsicCurvature<RealDataType, 3>,
                        tmpl::size_t<3>, Frame::Inertial>,
          gr::Tags::SpatialMetric<RealDataType, 3>,
          gr::Tags::InverseSpatialMetric<RealDataType, 3>,
          domain::Tags::Coordinates<3, Frame::Inertial>>,
      db::AddComputeTags<gr::Tags::Psi4RealCompute<Frame::Inertial>,
                         gr::Tags::Psi4ImagCompute<Frame::Inertial>>>(
      spatial_ricci, extrinsic_curvature, cov_deriv_extrinsic_curvature,
      spatial_metric, inv_spatial_metric, inertial_coords);
  const auto psi_4_real_expected = gr::psi_4_real(
      spatial_ricci, extrinsic_curvature, cov_deriv_extrinsic_curvature,
      spatial_metric, inv_spatial_metric, inertial_coords);
  CHECK_ITERABLE_APPROX((db::get<gr::Tags::Psi4Real<RealDataType>>(box)),
                        psi_4_real_expected);
  const auto psi_4_imag_expected = gr::psi_4_imag(
      spatial_ricci, extrinsic_curvature, cov_deriv_extrinsic_curvature,
      spatial_metric, inv_spatial_metric, inertial_coords);
  CHECK_ITERABLE_APPROX((db::get<gr::Tags::Psi4Imag<RealDataType>>(box)),
                        psi_4_imag_expected);
}

template <typename RealDataType, typename ComplexDataType>
void test_psi_4(const RealDataType& used_for_size_real,
                const ComplexDataType& /*used_for_size_complex*/) {
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> distribution(0.1, 3.0);
  const auto nn_generator = make_not_null(&generator);
  const auto nn_distribution = make_not_null(&distribution);

  const auto spatial_ricci =
      make_with_random_values<tnsr::ii<RealDataType, 3, Frame::Inertial>>(
          nn_generator, nn_distribution, used_for_size_real);
  const auto extrinsic_curvature =
      make_with_random_values<tnsr::ii<RealDataType, 3, Frame::Inertial>>(
          nn_generator, nn_distribution, used_for_size_real);
  const auto cov_deriv_extrinsic_curvature =
      make_with_random_values<tnsr::ijj<RealDataType, 3, Frame::Inertial>>(
          nn_generator, nn_distribution, used_for_size_real);
  const auto spatial_metric =
      TestHelpers::gr::random_spatial_metric<3, RealDataType, Frame::Inertial>(
          nn_generator, used_for_size_real);
  const auto inv_spatial_metric =
      determinant_and_inverse(spatial_metric).second;
  auto inertial_coords =
      make_with_random_values<tnsr::I<RealDataType, 3, Frame::Inertial>>(
          nn_generator, nn_distribution, used_for_size_real);

  const auto python_psi_4 = pypp::call<Scalar<ComplexDataType>>(
      "GeneralRelativity.Psi4", "psi_4", spatial_ricci, extrinsic_curvature,
      cov_deriv_extrinsic_curvature, spatial_metric, inv_spatial_metric,
      inertial_coords);
  const auto expected = gr::psi_4(spatial_ricci, extrinsic_curvature,
                                  cov_deriv_extrinsic_curvature, spatial_metric,
                                  inv_spatial_metric, inertial_coords);
  Approx local_approx = Approx::custom().epsilon(1e-13).scale(1.0);
  CHECK_ITERABLE_CUSTOM_APPROX(expected, python_psi_4, local_approx);
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
  // the transverse trace-free projection therefore gives Psi4 = 1/2 at every
  // azimuth, including on the x axis where the projected y direction is used
  // as a fallback.
  get<2, 2>(spatial_ricci) = DataVector{1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0};
  // The next two points check the historical Cartesian polarization basis on
  // the positive and negative z axes, which are grid lines in a filled Sphere
  // domain. The final point checks that same historical basis at the origin.
  // At all three points Psi4 = -1/2 + i for Ricci_xx = Ricci_xy = 1. The
  // imaginary part distinguishes the historical y polarization from its
  // negative, so this also checks that the fallback does not change the old
  // complex-Psi4 convention where the Gram-Schmidt construction is valid.
  get<0, 0>(spatial_ricci) = DataVector{0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0};
  get<0, 1>(spatial_ricci) = DataVector{0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0};
  auto inertial_coords =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(used_for_size,
                                                               0.0);
  const double inverse_sqrt_two = 1.0 / sqrt(2.0);
  get<0>(inertial_coords) =
      DataVector{1.0, inverse_sqrt_two, 0.0, -inverse_sqrt_two, 0.0, 0.0, 0.0};
  get<1>(inertial_coords) =
      DataVector{0.0, inverse_sqrt_two, 1.0, inverse_sqrt_two, 0.0, 0.0, 0.0};
  get<2>(inertial_coords) = DataVector{0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0};

  const auto result = gr::psi_4(spatial_ricci, extrinsic_curvature,
                                cov_deriv_extrinsic_curvature, spatial_metric,
                                inverse_spatial_metric, inertial_coords);
  auto expected = ComplexDataVector(7, 0.5);
  expected[4] = expected[5] = expected[6] = std::complex<double>{-0.5, 1.0};
  CHECK_ITERABLE_APPROX(get(result), expected);
  auto expected_real = DataVector(7, 0.5);
  expected_real[4] = expected_real[5] = expected_real[6] = -0.5;
  CHECK_ITERABLE_APPROX(
      get(gr::psi_4_real(spatial_ricci, extrinsic_curvature,
                         cov_deriv_extrinsic_curvature, spatial_metric,
                         inverse_spatial_metric, inertial_coords)),
      expected_real);
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
  // These Ricci components then give Psi4 = 1/2 + i.
  get<2, 2>(spatial_ricci) = 4.0;
  get<1, 2>(spatial_ricci) = sqrt(3.0);
  const auto result = gr::psi_4(spatial_ricci, extrinsic_curvature,
                                cov_deriv_extrinsic_curvature, spatial_metric,
                                inverse_spatial_metric, inertial_coords);
  const ComplexDataVector expected(1, std::complex<double>{0.5, 1.0});
  CHECK_ITERABLE_APPROX(get(result), expected);
  CHECK_ITERABLE_APPROX(
      get(gr::psi_4_real(spatial_ricci, extrinsic_curvature,
                         cov_deriv_extrinsic_curvature, spatial_metric,
                         inverse_spatial_metric, inertial_coords)),
      DataVector(1, 0.5));
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

  ComplexDataVector expected(used_for_size.size());
  for (size_t p = 0; p < used_for_size.size(); ++p) {
    const double y = get<1>(coords)[p];
    const double z = get<2>(coords)[p];
    const double transverse_radius = hypot(y, z);
    const double radius = hypot(1.0, transverse_radius);
    const double orientation = z > 0.0 ? 1.0 : -1.0;
    // Analytic Cartesian polarizations avoid subtracting nearly parallel
    // vectors. With K_ij = 0 the trace-free projection drops out when
    // contracted with this transverse complex null vector.
    const std::complex<double> m_x{transverse_radius / radius, 0.0};
    const std::complex<double> m_y{-y / (radius * transverse_radius),
                                   -orientation * z / transverse_radius};
    const std::complex<double> m_z{-z / (radius * transverse_radius),
                                   orientation * y / transverse_radius};
    expected[p] = -0.5 * (0.4 * m_x * m_x + 0.4 * m_x * m_y + 0.6 * m_x * m_z -
                          0.2 * m_y * m_y + m_y * m_z + m_z * m_z);
  }
  CHECK_ITERABLE_APPROX(
      get(gr::psi_4(spatial_ricci, extrinsic_curvature,
                    cov_deriv_extrinsic_curvature, spatial_metric,
                    inverse_spatial_metric, coords)),
      expected);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.GeneralRelativity.Psi4",
                  "[Unit][PointwiseFunctions]") {
  pypp::SetupLocalPythonEnvironment local_python_env("PointwiseFunctions/");

  const size_t size = 5;
  const DataVector used_for_size_real_dv =
      DataVector(size, std::numeric_limits<double>::signaling_NaN());
  const ComplexDataVector used_for_size_complex_dv =
      ComplexDataVector(size, std::numeric_limits<double>::signaling_NaN());
  test_psi_4(used_for_size_real_dv, used_for_size_complex_dv);
  test_compute_item_in_databox(used_for_size_real_dv);
  test_polarization_basis_on_coordinate_planes();
  test_polarization_basis_with_curved_metric();
  test_nearly_degenerate_polarization_basis();
}
