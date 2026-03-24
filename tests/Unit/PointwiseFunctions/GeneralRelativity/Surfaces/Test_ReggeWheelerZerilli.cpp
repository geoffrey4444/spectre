// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>

#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/StrahlkorperFunctions.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/WrappedGr.hpp"
#include "PointwiseFunctions/GeneralRelativity/Surfaces/ReggeWheelerZerilli.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {

size_t goldberg_index(const size_t l_max, const size_t l, const int m) {
  return static_cast<size_t>(
      static_cast<int>(square(l_max + 1) + square(l) + l) + m -
      static_cast<int>(square(l_max + 1)));
}

void set_mode(const gsl::not_null<ComplexModalVector*> data, const size_t l_max,
              const size_t l, const int m, const std::complex<double>& value) {
  (*data)[goldberg_index(l_max, l, m)] = value;
}

std::complex<double> mode(const ComplexModalVector& data, const size_t l_max,
                          const size_t l, const int m) {
  return data[goldberg_index(l_max, l, m)];
}

void check_complex_approx(const std::complex<double>& actual,
                          const std::complex<double>& expected) {
  CHECK(real(actual) == approx(real(expected)));
  CHECK(imag(actual) == approx(imag(expected)));
}

void test_regge_wheeler_zerilli_moncrief() {
  const size_t l_max = 3;
  const double radius = 10.0;
  const size_t number_of_modes = square(l_max + 1);

  ComplexModalVector h_t{number_of_modes, 0.0};
  ComplexModalVector dr_h_t{number_of_modes, 0.0};
  ComplexModalVector dt_h_r{number_of_modes, 0.0};
  ComplexModalVector h_rr{number_of_modes, 0.0};
  ComplexModalVector q_r{number_of_modes, 0.0};
  ComplexModalVector k{number_of_modes, 0.0};
  ComplexModalVector dr_k{number_of_modes, 0.0};
  ComplexModalVector g{number_of_modes, 0.0};
  ComplexModalVector dr_g{number_of_modes, 0.0};

  const std::complex<double> h_t_22{1.0, -0.5};
  const std::complex<double> dr_h_t_22{-0.25, 0.75};
  const std::complex<double> dt_h_r_22{0.8, 0.1};
  const std::complex<double> h_rr_22{0.5, -0.3};
  const std::complex<double> q_r_22{-0.4, 0.2};
  const std::complex<double> k_22{0.3, 0.7};
  const std::complex<double> dr_k_22{-0.1, 0.4};
  const std::complex<double> g_22{0.2, -0.6};
  const std::complex<double> dr_g_22{0.05, 0.15};

  set_mode(make_not_null(&h_t), l_max, 2, 2, h_t_22);
  set_mode(make_not_null(&dr_h_t), l_max, 2, 2, dr_h_t_22);
  set_mode(make_not_null(&dt_h_r), l_max, 2, 2, dt_h_r_22);
  set_mode(make_not_null(&h_rr), l_max, 2, 2, h_rr_22);
  set_mode(make_not_null(&q_r), l_max, 2, 2, q_r_22);
  set_mode(make_not_null(&k), l_max, 2, 2, k_22);
  set_mode(make_not_null(&dr_k), l_max, 2, 2, dr_k_22);
  set_mode(make_not_null(&g), l_max, 2, 2, g_22);
  set_mode(make_not_null(&dr_g), l_max, 2, 2, dr_g_22);

  const std::complex<double> h_t_31{-0.4, 0.9};
  const std::complex<double> dr_h_t_31{0.3, -0.2};
  const std::complex<double> dt_h_r_31{-0.8, 0.5};
  const std::complex<double> h_rr_31{1.1, 0.4};
  const std::complex<double> q_r_31{0.6, -0.7};
  const std::complex<double> k_31{-0.2, 0.1};
  const std::complex<double> dr_k_31{0.45, -0.35};
  const std::complex<double> g_31{0.15, 0.2};
  const std::complex<double> dr_g_31{-0.05, 0.08};

  set_mode(make_not_null(&h_t), l_max, 3, 1, h_t_31);
  set_mode(make_not_null(&dr_h_t), l_max, 3, 1, dr_h_t_31);
  set_mode(make_not_null(&dt_h_r), l_max, 3, 1, dt_h_r_31);
  set_mode(make_not_null(&h_rr), l_max, 3, 1, h_rr_31);
  set_mode(make_not_null(&q_r), l_max, 3, 1, q_r_31);
  set_mode(make_not_null(&k), l_max, 3, 1, k_31);
  set_mode(make_not_null(&dr_k), l_max, 3, 1, dr_k_31);
  set_mode(make_not_null(&g), l_max, 3, 1, g_31);
  set_mode(make_not_null(&dr_g), l_max, 3, 1, dr_g_31);

  // Modes with l < 2 should be ignored.
  set_mode(make_not_null(&h_t), l_max, 1, 1, {10.0, 20.0});
  set_mode(make_not_null(&dr_h_t), l_max, 1, 1, {30.0, -40.0});
  set_mode(make_not_null(&dt_h_r), l_max, 1, 1, {-50.0, 60.0});
  set_mode(make_not_null(&h_rr), l_max, 1, 1, {70.0, -80.0});
  set_mode(make_not_null(&q_r), l_max, 1, 1, {-90.0, 100.0});
  set_mode(make_not_null(&k), l_max, 1, 1, {110.0, 120.0});
  set_mode(make_not_null(&dr_k), l_max, 1, 1, {-130.0, 140.0});
  set_mode(make_not_null(&g), l_max, 1, 1, {150.0, -160.0});
  set_mode(make_not_null(&dr_g), l_max, 1, 1, {170.0, 180.0});

  const auto rwz = gr::surfaces::regge_wheeler_zerilli_moncrief(
      h_t, dr_h_t, dt_h_r, h_rr, q_r, k, dr_k, g, dr_g, l_max, radius);

  const std::complex<double> i{0.0, 1.0};

  const auto p_r_22 = q_r_22 - 0.5 * square(radius) * dr_g_22;
  const auto z_r_22 = h_rr_22 - radius * dr_k_22 - 3.0 * radius * dr_g_22 -
                      2.0 * p_r_22 / radius;
  const auto k_invariant_22 = k_22 + 3.0 * g_22 - 2.0 * p_r_22 / radius;
  const auto expected_phi_minus_22 =
      (radius * (dt_h_r_22 - dr_h_t_22) + 2.0 * h_t_22) / 4.0;
  const auto expected_phi_plus_22 =
      radius * (2.0 * z_r_22 + 4.0 * k_invariant_22) / 24.0;
  const auto expected_rh_22 =
      sqrt(24.0) * (expected_phi_plus_22 + i * expected_phi_minus_22);

  check_complex_approx(mode(rwz.phi_plus, l_max, 2, 2), expected_phi_plus_22);
  check_complex_approx(mode(rwz.phi_minus, l_max, 2, 2), expected_phi_minus_22);
  check_complex_approx(mode(rwz.r_times_strain, l_max, 2, 2), expected_rh_22);
  check_complex_approx(mode(rwz.phi_plus, l_max, 2, -2),
                       conj(expected_phi_plus_22));
  check_complex_approx(mode(rwz.phi_minus, l_max, 2, -2),
                       conj(expected_phi_minus_22));
  check_complex_approx(mode(rwz.r_times_strain, l_max, 2, -2),
                       conj(expected_rh_22));

  const auto p_r_31 = q_r_31 - 0.5 * square(radius) * dr_g_31;
  const auto z_r_31 = h_rr_31 - radius * dr_k_31 - 6.0 * radius * dr_g_31 -
                      2.0 * p_r_31 / radius;
  const auto k_invariant_31 = k_31 + 6.0 * g_31 - 2.0 * p_r_31 / radius;
  const auto expected_phi_minus_31 =
      (radius * (dt_h_r_31 - dr_h_t_31) + 2.0 * h_t_31) / 10.0;
  const auto expected_phi_plus_31 =
      radius * (2.0 * z_r_31 + 10.0 * k_invariant_31) / 120.0;
  const auto expected_rh_31 =
      sqrt(120.0) * (expected_phi_plus_31 + i * expected_phi_minus_31);

  check_complex_approx(mode(rwz.phi_plus, l_max, 3, 1), expected_phi_plus_31);
  check_complex_approx(mode(rwz.phi_minus, l_max, 3, 1), expected_phi_minus_31);
  check_complex_approx(mode(rwz.r_times_strain, l_max, 3, 1), expected_rh_31);
  check_complex_approx(mode(rwz.phi_plus, l_max, 3, -1),
                       -conj(expected_phi_plus_31));
  check_complex_approx(mode(rwz.phi_minus, l_max, 3, -1),
                       -conj(expected_phi_minus_31));
  check_complex_approx(mode(rwz.r_times_strain, l_max, 3, -1),
                       -conj(expected_rh_31));

  CHECK(mode(rwz.phi_plus, l_max, 1, 1) == std::complex<double>{0.0, 0.0});
  CHECK(mode(rwz.phi_minus, l_max, 1, -1) == std::complex<double>{0.0, 0.0});
  CHECK(mode(rwz.r_times_strain, l_max, 0, 0) ==
        std::complex<double>{0.0, 0.0});
}

void test_regge_wheeler_zerilli_from_gh_vars_minkowski() {
  const size_t l_max = 4;
  const double radius = 3.0;
  const std::array<double, 3> center{{0.1, -0.2, 0.3}};
  const ylm::Strahlkorper<Frame::Inertial> strahlkorper{l_max, l_max, radius,
                                                        center};
  const auto coords = ylm::cartesian_coords(strahlkorper);
  const size_t number_of_points = get<0>(coords).size();

  tnsr::aa<DataVector, 3, Frame::Inertial> spacetime_metric{number_of_points,
                                                            0.0};
  tnsr::aa<DataVector, 3, Frame::Inertial> pi{number_of_points, 0.0};
  tnsr::iaa<DataVector, 3, Frame::Inertial> phi{number_of_points, 0.0};

  get<0, 0>(spacetime_metric) = -1.0;
  for (size_t i = 0; i < 3; ++i) {
    spacetime_metric.get(i + 1, i + 1) = 1.0;
  }

  const auto rwz = gr::surfaces::regge_wheeler_zerilli_moncrief_from_gh_vars(
      spacetime_metric, pi, phi, coords, strahlkorper.ylm_spherepack(), center,
      radius);

  for (const auto& mode_value : rwz.phi_plus) {
    CHECK(mode_value == std::complex<double>{0.0, 0.0});
  }
  for (const auto& mode_value : rwz.phi_minus) {
    CHECK(mode_value == std::complex<double>{0.0, 0.0});
  }
  for (const auto& mode_value : rwz.r_times_strain) {
    CHECK(mode_value == std::complex<double>{0.0, 0.0});
  }
}

void test_extraction_sphere_metadata_from_gh_vars_minkowski() {
  const size_t l_max = 6;
  const double radius = 7.5;
  const std::array<double, 3> center{{0.0, 0.0, 0.0}};
  const ylm::Strahlkorper<Frame::Inertial> strahlkorper{l_max, l_max, radius,
                                                        center};
  const size_t number_of_points = strahlkorper.ylm_spherepack().physical_size();

  tnsr::aa<DataVector, 3, Frame::Inertial> spacetime_metric{number_of_points,
                                                            0.0};
  get<0, 0>(spacetime_metric) = -1.0;
  for (size_t i = 0; i < 3; ++i) {
    spacetime_metric.get(i + 1, i + 1) = 1.0;
  }

  const auto metadata = gr::surfaces::extraction_sphere_metadata_from_gh_vars(
      spacetime_metric, strahlkorper);
  CHECK(metadata.average_lapse == approx(1.0));
  CHECK(metadata.areal_radius == approx(radius));
}

void test_psi_4_modes_from_tensors_minkowski() {
  const size_t l_max = 5;
  const double radius = 6.0;
  const std::array<double, 3> center{{0.2, -0.3, 0.4}};
  const ylm::Strahlkorper<Frame::Inertial> strahlkorper{l_max, l_max, radius,
                                                        center};
  const auto coords = ylm::cartesian_coords(strahlkorper);
  const size_t number_of_points = get<0>(coords).size();

  tnsr::aa<DataVector, 3, Frame::Inertial> spacetime_metric{number_of_points,
                                                            0.0};
  get<0, 0>(spacetime_metric) = -1.0;
  for (size_t i = 0; i < 3; ++i) {
    spacetime_metric.get(i + 1, i + 1) = 1.0;
  }

  const tnsr::ii<DataVector, 3, Frame::Inertial> spatial_ricci{number_of_points,
                                                               0.0};
  const tnsr::ii<DataVector, 3, Frame::Inertial> extrinsic_curvature{
      number_of_points, 0.0};
  const tnsr::ijj<DataVector, 3, Frame::Inertial> cov_deriv_extrinsic_curvature{
      number_of_points, 0.0};

  const auto r_times_psi_4 = gr::surfaces::psi_4_modes_from_tensors(
      spacetime_metric, spatial_ricci, extrinsic_curvature,
      cov_deriv_extrinsic_curvature, coords, strahlkorper.ylm_spherepack(),
      center, radius);

  for (const auto& mode_value : r_times_psi_4) {
    CHECK(mode_value == std::complex<double>{0.0, 0.0});
  }
}

void test_regge_wheeler_zerilli_from_gh_vars_kerr_schild_schwarzschild() {
  const size_t l_max = 8;
  const double radius = 10.0;
  const std::array<double, 3> center{{0.0, 0.0, 0.0}};
  const std::array<double, 3> spin{{0.0, 0.0, 0.0}};
  const std::array<double, 3> velocity{{0.0, 0.0, 0.0}};
  const ylm::Strahlkorper<Frame::Inertial> strahlkorper{l_max, l_max, radius,
                                                        center};
  const auto coords = ylm::cartesian_coords(strahlkorper);
  const auto solution = gh::Solutions::WrappedGr<gr::Solutions::KerrSchild>{
      1.0, spin, center, velocity};
  const auto gh_vars = solution.variables(
      coords, 0.0,
      tmpl::list<gr::Tags::SpacetimeMetric<DataVector, 3>,
                 gh::Tags::Pi<DataVector, 3>, gh::Tags::Phi<DataVector, 3>>{});

  const auto rwz = gr::surfaces::regge_wheeler_zerilli_moncrief_from_gh_vars(
      get<gr::Tags::SpacetimeMetric<DataVector, 3>>(gh_vars),
      get<gh::Tags::Pi<DataVector, 3>>(gh_vars),
      get<gh::Tags::Phi<DataVector, 3>>(gh_vars), coords,
      strahlkorper.ylm_spherepack(), center, radius);

  const auto check_modes = [](const ComplexModalVector& modes) {
    for (const auto& mode_value : modes) {
      CHECK(std::isfinite(real(mode_value)));
      CHECK(std::isfinite(imag(mode_value)));
      CHECK(abs(mode_value) < 1.0e-11);
    }
  };
  check_modes(rwz.phi_plus);
  check_modes(rwz.phi_minus);
  check_modes(rwz.r_times_strain);
}

}  // namespace

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.GeneralRelativity.Surfaces."
    "ReggeWheelerZerilli",
    "[PointwiseFunctions][Unit]") {
  test_regge_wheeler_zerilli_moncrief();
  test_regge_wheeler_zerilli_from_gh_vars_minkowski();
  test_regge_wheeler_zerilli_from_gh_vars_kerr_schild_schwarzschild();
  test_extraction_sphere_metadata_from_gh_vars_minkowski();
  test_psi_4_modes_from_tensors_minkowski();
}
