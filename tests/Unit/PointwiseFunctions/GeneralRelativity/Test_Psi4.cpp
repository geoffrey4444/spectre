// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
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
#include "PointwiseFunctions/GeneralRelativity/Psi4Real.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "PointwiseFunctions/GeneralRelativity/WeylPropagating.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace {
template <typename RealDataType>
void test_compute_item_in_databox(const RealDataType& used_for_size_real) {
  TestHelpers::db::test_compute_tag<gr::Tags::Psi4RealCompute<Frame::Inertial>>(
      "Psi4Real");

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
      db::AddComputeTags<gr::Tags::Psi4RealCompute<Frame::Inertial>>>(
      spatial_ricci, extrinsic_curvature, cov_deriv_extrinsic_curvature,
      spatial_metric, inv_spatial_metric, inertial_coords);
  const auto expected = gr::psi_4_real(
      spatial_ricci, extrinsic_curvature, cov_deriv_extrinsic_curvature,
      spatial_metric, inv_spatial_metric, inertial_coords);
  CHECK_ITERABLE_APPROX((db::get<gr::Tags::Psi4Real<RealDataType>>(box)),
                        expected);
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
  auto extrinsic_curvature =
      make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(used_for_size,
                                                                0.0);
  auto cov_deriv_extrinsic_curvature =
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
  spatial_ricci.get(2, 2) = DataVector{1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0};
  // The next two points check the historical Cartesian polarization basis on
  // the positive and negative z axes, which are grid lines in a filled Sphere
  // domain. The final point checks that same historical basis at the origin.
  // At all three points Psi4 = -1/2 + i for Ricci_xx = Ricci_xy = 1. The
  // imaginary part distinguishes the historical y polarization from its
  // negative, so this also checks that the fallback does not change the old
  // complex-Psi4 convention where the Gram-Schmidt construction is valid.
  spatial_ricci.get(0, 0) = DataVector{0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0};
  spatial_ricci.get(0, 1) = DataVector{0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0};
  auto inertial_coords =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(used_for_size,
                                                               0.0);
  const double inverse_sqrt_two = 1.0 / sqrt(2.0);
  inertial_coords.get(0) =
      DataVector{1.0, inverse_sqrt_two, 0.0, -inverse_sqrt_two, 0.0, 0.0, 0.0};
  inertial_coords.get(1) =
      DataVector{0.0, inverse_sqrt_two, 1.0, inverse_sqrt_two, 0.0, 0.0, 0.0};
  inertial_coords.get(2) = DataVector{0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0};

  const auto result = gr::psi_4(spatial_ricci, extrinsic_curvature,
                                cov_deriv_extrinsic_curvature, spatial_metric,
                                inverse_spatial_metric, inertial_coords);
  auto expected = ComplexDataVector(7, 0.5);
  expected[4] = expected[5] = expected[6] = std::complex<double>{-0.5, 1.0};
  CHECK_ITERABLE_APPROX(get(result), expected);
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
}
