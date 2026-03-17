// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/Surfaces/ReggeWheelerZerilli.hpp"

#include <cmath>
#include <complex>
#include <cstddef>

#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"

namespace gr::surfaces {
namespace {

void assert_size(const ComplexModalVector& data, const size_t expected_size,
                 const char* const name) {
  ASSERT(data.size() == expected_size,
         "Expected " << name << " to have size " << expected_size << " but got "
                     << data.size());
}

size_t goldberg_index(const size_t l_max, const size_t l, const int m) {
  return static_cast<size_t>(
      static_cast<int>(square(l_max + 1) + square(l) + l) + m -
      static_cast<int>(square(l_max + 1)));
}

std::complex<double> goldberg_mode(const ComplexModalVector& data,
                                   const size_t l_max, const size_t l,
                                   const int m) {
  return data[goldberg_index(l_max, l, m)];
}

void set_goldberg_mode(const gsl::not_null<ComplexModalVector*> data,
                       const size_t l_max, const size_t l, const int m,
                       const std::complex<double>& value) {
  (*data)[goldberg_index(l_max, l, m)] = value;
}

void fill_negative_m_modes(const gsl::not_null<ComplexModalVector*> data,
                           const size_t l_max) {
  for (size_t l = 0; l <= l_max; ++l) {
    for (int m = 1; m <= static_cast<int>(l); ++m) {
      const double sign = m % 2 == 0 ? 1.0 : -1.0;
      set_goldberg_mode(data, l_max, l, -m,
                        sign * conj(goldberg_mode(*data, l_max, l, m)));
    }
  }
}

}  // namespace

ReggeWheelerZerilli::ReggeWheelerZerilli(const size_t l_max)
    : phi_plus(square(l_max + 1), 0.0),
      phi_minus(square(l_max + 1), 0.0),
      r_times_strain(square(l_max + 1), 0.0) {}

void regge_wheeler_zerilli_moncrief(
    const gsl::not_null<ReggeWheelerZerilli*> rwz_quantities,
    const ComplexModalVector& h_t, const ComplexModalVector& dr_h_t,
    const ComplexModalVector& dt_h_r, const ComplexModalVector& h_rr,
    const ComplexModalVector& q_r, const ComplexModalVector& k,
    const ComplexModalVector& dr_k, const ComplexModalVector& g,
    const ComplexModalVector& dr_g, const size_t l_max,
    const double extraction_radius) {
  ASSERT(
      extraction_radius > 0.0,
      "The extraction radius must be positive, but is " << extraction_radius);
  const size_t number_of_modes = square(l_max + 1);
  assert_size(h_t, number_of_modes, "h_t");
  assert_size(dr_h_t, number_of_modes, "dr_h_t");
  assert_size(dt_h_r, number_of_modes, "dt_h_r");
  assert_size(h_rr, number_of_modes, "h_rr");
  assert_size(q_r, number_of_modes, "q_r");
  assert_size(k, number_of_modes, "k");
  assert_size(dr_k, number_of_modes, "dr_k");
  assert_size(g, number_of_modes, "g");
  assert_size(dr_g, number_of_modes, "dr_g");

  rwz_quantities->phi_plus = ComplexModalVector{number_of_modes, 0.0};
  rwz_quantities->phi_minus = ComplexModalVector{number_of_modes, 0.0};
  rwz_quantities->r_times_strain = ComplexModalVector{number_of_modes, 0.0};

  const std::complex<double> i{0.0, 1.0};
  for (size_t l = 2; l <= l_max; ++l) {
    const double lambda = static_cast<double>((l - 1) * (l + 2));
    const double l_l_plus_1 = static_cast<double>(l * (l + 1));
    const double strain_prefactor =
        sqrt(static_cast<double>((l - 1) * l * (l + 1) * (l + 2)));
    for (int m = 0; m <= static_cast<int>(l); ++m) {
      const auto p_r =
          goldberg_mode(q_r, l_max, l, m) -
          0.5 * square(extraction_radius) * goldberg_mode(dr_g, l_max, l, m);
      const auto z_r = goldberg_mode(h_rr, l_max, l, m) -
                       extraction_radius * goldberg_mode(dr_k, l_max, l, m) -
                       0.5 * extraction_radius * l_l_plus_1 *
                           goldberg_mode(dr_g, l_max, l, m) -
                       2.0 * p_r / extraction_radius;
      const auto k_invariant =
          goldberg_mode(k, l_max, l, m) +
          0.5 * l_l_plus_1 * goldberg_mode(g, l_max, l, m) -
          2.0 * p_r / extraction_radius;
      const auto phi_minus =
          (extraction_radius * (goldberg_mode(dt_h_r, l_max, l, m) -
                                goldberg_mode(dr_h_t, l_max, l, m)) +
           2.0 * goldberg_mode(h_t, l_max, l, m)) /
          lambda;
      const auto phi_plus = extraction_radius *
                            (2.0 * z_r + lambda * k_invariant) /
                            (lambda * l_l_plus_1);
      set_goldberg_mode(make_not_null(&rwz_quantities->phi_plus), l_max, l, m,
                        phi_plus);
      set_goldberg_mode(make_not_null(&rwz_quantities->phi_minus), l_max, l, m,
                        phi_minus);
      set_goldberg_mode(make_not_null(&rwz_quantities->r_times_strain), l_max,
                        l, m, strain_prefactor * (phi_plus + i * phi_minus));
    }
  }

  fill_negative_m_modes(make_not_null(&rwz_quantities->phi_plus), l_max);
  fill_negative_m_modes(make_not_null(&rwz_quantities->phi_minus), l_max);
  fill_negative_m_modes(make_not_null(&rwz_quantities->r_times_strain), l_max);
}

ReggeWheelerZerilli regge_wheeler_zerilli_moncrief(
    const ComplexModalVector& h_t, const ComplexModalVector& dr_h_t,
    const ComplexModalVector& dt_h_r, const ComplexModalVector& h_rr,
    const ComplexModalVector& q_r, const ComplexModalVector& k,
    const ComplexModalVector& dr_k, const ComplexModalVector& g,
    const ComplexModalVector& dr_g, const size_t l_max,
    const double extraction_radius) {
  ReggeWheelerZerilli rwz_quantities{l_max};
  regge_wheeler_zerilli_moncrief(make_not_null(&rwz_quantities), h_t, dr_h_t,
                                 dt_h_r, h_rr, q_r, k, dr_k, g, dr_g, l_max,
                                 extraction_radius);
  return rwz_quantities;
}

}  // namespace gr::surfaces
