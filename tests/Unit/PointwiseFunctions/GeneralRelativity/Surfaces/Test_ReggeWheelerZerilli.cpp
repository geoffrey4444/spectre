// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <complex>
#include <cstddef>

#include "DataStructures/ComplexModalVector.hpp"
#include "PointwiseFunctions/GeneralRelativity/Surfaces/ReggeWheelerZerilli.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"

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

}  // namespace

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.GeneralRelativity.Surfaces."
    "ReggeWheelerZerilli",
    "[PointwiseFunctions][Unit]") {
  test_regge_wheeler_zerilli_moncrief();
}
