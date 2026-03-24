// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/Surfaces/ReggeWheelerZerilli.hpp"

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/SimpleSparseMatrix.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackIterator.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/StrahlkorperFunctions.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/TensorYlm.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/TensorYlmCartToSphere.hpp"
#include "PointwiseFunctions/GeneralRelativity/Lapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/ProjectionOperators.hpp"
#include "PointwiseFunctions/GeneralRelativity/Shift.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Surfaces/AreaElement.hpp"
#include "PointwiseFunctions/GeneralRelativity/Surfaces/SurfaceIntegralOfScalar.hpp"
#include "PointwiseFunctions/GeneralRelativity/WeylPropagating.hpp"
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

std::complex<double> standard_mode_from_spherepack(const DataVector& data,
                                                   const size_t l_max,
                                                   const size_t l,
                                                   const int m) {
  ylm::SpherepackIterator iterator(l_max, l_max, 1, false);
  const auto a_index =
      iterator.set(l, static_cast<size_t>(m),
                   ylm::SpherepackIterator::CoefficientArray::a)();
  const std::complex<double> spherepack_mode =
      m == 0 ? std::complex<double>{data[a_index], 0.0}
             : std::complex<double>{
                   data[a_index],
                   data[iterator.set(
                       l, static_cast<size_t>(m),
                       ylm::SpherepackIterator::CoefficientArray::b)()]};
  if (not(std::isfinite(real(spherepack_mode)) and
          std::isfinite(imag(spherepack_mode)))) {
    return {0.0, 0.0};
  }
  const double sign = m % 2 == 0 ? 1.0 : -1.0;
  return sign * sqrt(M_PI / 2.0) * spherepack_mode;
}

template <typename TensorType>
TensorType cartesian_to_spherical_tensor_modes(const TensorType& nodal_tensor,
                                               const ylm::Spherepack& ylm) {
  const size_t spectral_size =
      ylm::SpherepackIterator(ylm.l_max(), ylm.m_max(), 1, false)
          .spherepack_array_size();
  TensorType cartesian_modes{spectral_size, 0.0};
  TensorType spherical_modes{spectral_size, 0.0};

  for (size_t storage_index = 0; storage_index < nodal_tensor.size();
       ++storage_index) {
    cartesian_modes[storage_index] =
        ylm.phys_to_spec(nodal_tensor[storage_index]);
  }
  ylm::SpherepackIterator iterator(ylm.l_max(), ylm.m_max(), 1, false);
  for (size_t storage_index = 0; storage_index < cartesian_modes.size();
       ++storage_index) {
    for (size_t offset = 0; offset < spectral_size; ++offset) {
      if (not iterator.compact_index(offset).has_value()) {
        cartesian_modes[storage_index][offset] = 0.0;
      }
    }
  }

  SimpleSparseMatrix cart_to_sphere_matrix{};
  ylm::TensorYlm::fill_cart_to_sphere<typename TensorType::structure>(
      make_not_null(&cart_to_sphere_matrix), ylm.l_max(),
      ylm::TensorYlm::CoefficientNormalization::Spherepack);

  std::vector<double> flattened_cartesian_modes(cartesian_modes.size() *
                                                spectral_size);
  std::vector<double> flattened_spherical_modes(
      spherical_modes.size() * spectral_size, 0.0);
  for (size_t storage_index = 0; storage_index < cartesian_modes.size();
       ++storage_index) {
    for (size_t mode_index = 0; mode_index < spectral_size; ++mode_index) {
      flattened_cartesian_modes[storage_index * spectral_size + mode_index] =
          cartesian_modes[storage_index][mode_index];
    }
  }

  gsl::span<double> spherical_modes_span{flattened_spherical_modes};
  const gsl::span<double> cartesian_modes_span{flattened_cartesian_modes};
  cart_to_sphere_matrix.increment_multiply_on_right(
      make_not_null(&spherical_modes_span), 0, 1, cartesian_modes_span, 0, 1);
  for (size_t storage_index = 0; storage_index < spherical_modes.size();
       ++storage_index) {
    for (size_t mode_index = 0; mode_index < spectral_size; ++mode_index) {
      spherical_modes[storage_index][mode_index] =
          flattened_spherical_modes[storage_index * spectral_size + mode_index];
    }
  }
  return spherical_modes;
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

void regge_wheeler_zerilli_moncrief_from_gh_vars(
    const gsl::not_null<ReggeWheelerZerilli*> rwz_quantities,
    const tnsr::aa<DataVector, 3, Frame::Inertial>& spacetime_metric,
    const tnsr::aa<DataVector, 3, Frame::Inertial>& pi,
    const tnsr::iaa<DataVector, 3, Frame::Inertial>& phi,
    const tnsr::I<DataVector, 3, Frame::Inertial>& inertial_coords,
    const ylm::Spherepack& ylm_spherepack, const std::array<double, 3>& center,
    const double extraction_radius) {
  ASSERT(
      extraction_radius > 0.0,
      "The extraction radius must be positive, but is " << extraction_radius);
  ASSERT(get<0>(inertial_coords).size() == ylm_spherepack.physical_size(),
         "Expected one extraction sphere worth of nodal points ("
             << ylm_spherepack.physical_size() << "), but received "
             << get<0>(inertial_coords).size());

  const size_t l_max = ylm_spherepack.l_max();
  const size_t number_of_points = get<0>(inertial_coords).size();
  const size_t number_of_modes = square(l_max + 1);
  bool is_flat_background_data = true;
  for (size_t a = 0; a < 4; ++a) {
    for (size_t b = 0; b < 4; ++b) {
      const double expected_spacetime_metric =
          a == b ? (a == 0 ? -1.0 : 1.0) : 0.0;
      for (size_t s = 0; s < number_of_points; ++s) {
        is_flat_background_data =
            is_flat_background_data and
            spacetime_metric.get(a, b)[s] == expected_spacetime_metric and
            pi.get(a, b)[s] == 0.0;
      }
    }
  }
  for (size_t i = 0; i < 3; ++i) {
    for (size_t a = 0; a < 4; ++a) {
      for (size_t b = 0; b < 4; ++b) {
        for (size_t s = 0; s < number_of_points; ++s) {
          is_flat_background_data =
              is_flat_background_data and phi.get(i, a, b)[s] == 0.0;
        }
      }
    }
  }
  if (is_flat_background_data) {
    *rwz_quantities = ReggeWheelerZerilli{l_max};
    return;
  }

  tnsr::i<DataVector, 3, Frame::Inertial> radial_unit_vector{number_of_points};
  tnsr::i<DataVector, 3, Frame::Inertial> shift_vector{number_of_points, 0.0};
  tnsr::i<DataVector, 3, Frame::Inertial> shift_prime{number_of_points, 0.0};
  tnsr::ii<DataVector, 3, Frame::Inertial> spatial_metric_perturbation{
      number_of_points, 0.0};
  tnsr::ii<DataVector, 3, Frame::Inertial> dt_spatial_metric{number_of_points,
                                                             0.0};
  tnsr::ii<DataVector, 3, Frame::Inertial> dr_spatial_metric{number_of_points,
                                                             0.0};

  for (size_t i = 0; i < 3; ++i) {
    radial_unit_vector.get(i) =
        (inertial_coords.get(i) - center[i]) / extraction_radius;
    shift_vector.get(i) = spacetime_metric.get(i + 1, 0);
    for (size_t j = i; j < 3; ++j) {
      spatial_metric_perturbation.get(i, j) =
          spacetime_metric.get(i + 1, j + 1);
      if (i == j) {
        spatial_metric_perturbation.get(i, j) -= 1.0;
      }
      dt_spatial_metric.get(i, j) = -pi.get(i + 1, j + 1);
      dr_spatial_metric.get(i, j) = 0.0;
      for (size_t k = 0; k < 3; ++k) {
        dr_spatial_metric.get(i, j) +=
            radial_unit_vector.get(k) * phi.get(k, i + 1, j + 1);
      }
    }
    shift_prime.get(i) = 0.0;
    for (size_t k = 0; k < 3; ++k) {
      shift_prime.get(i) += radial_unit_vector.get(k) * phi.get(k, i + 1, 0);
    }
  }
  bool all_perturbations_zero = true;
  for (const auto& component : shift_vector) {
    for (const double value : component) {
      all_perturbations_zero = all_perturbations_zero and value == 0.0;
    }
  }
  for (const auto& component : shift_prime) {
    for (const double value : component) {
      all_perturbations_zero = all_perturbations_zero and value == 0.0;
    }
  }
  for (const auto& component : spatial_metric_perturbation) {
    for (const double value : component) {
      all_perturbations_zero = all_perturbations_zero and value == 0.0;
    }
  }
  for (const auto& component : dt_spatial_metric) {
    for (const double value : component) {
      all_perturbations_zero = all_perturbations_zero and value == 0.0;
    }
  }
  for (const auto& component : dr_spatial_metric) {
    for (const double value : component) {
      all_perturbations_zero = all_perturbations_zero and value == 0.0;
    }
  }
  if (all_perturbations_zero) {
    *rwz_quantities = ReggeWheelerZerilli{l_max};
    return;
  }

  const auto shift_modes =
      cartesian_to_spherical_tensor_modes(shift_vector, ylm_spherepack);
  const auto shift_prime_modes =
      cartesian_to_spherical_tensor_modes(shift_prime, ylm_spherepack);
  const auto metric_modes = cartesian_to_spherical_tensor_modes(
      spatial_metric_perturbation, ylm_spherepack);
  const auto dt_metric_modes =
      cartesian_to_spherical_tensor_modes(dt_spatial_metric, ylm_spherepack);
  const auto dr_metric_modes =
      cartesian_to_spherical_tensor_modes(dr_spatial_metric, ylm_spherepack);

  ComplexModalVector h_t{number_of_modes, 0.0};
  ComplexModalVector dr_h_t{number_of_modes, 0.0};
  ComplexModalVector dt_h_r{number_of_modes, 0.0};
  ComplexModalVector h_rr{number_of_modes, 0.0};
  ComplexModalVector q_r{number_of_modes, 0.0};
  ComplexModalVector k{number_of_modes, 0.0};
  ComplexModalVector dr_k{number_of_modes, 0.0};
  ComplexModalVector g{number_of_modes, 0.0};
  ComplexModalVector dr_g{number_of_modes, 0.0};

  const std::complex<double> i{0.0, 1.0};
  for (size_t l = 2; l <= l_max; ++l) {
    const double vector_prefactor =
        1.0 / sqrt(2.0 * static_cast<double>(l * (l + 1)));
    const double tensor_prefactor =
        1.0 / sqrt(static_cast<double>((l - 1) * l * (l + 1) * (l + 2)));
    for (int m = 0; m <= static_cast<int>(l); ++m) {
      const auto v_m =
          standard_mode_from_spherepack(shift_modes.get(1), l_max, l, m);
      const auto v_mbar =
          standard_mode_from_spherepack(shift_modes.get(2), l_max, l, m);
      const auto vp_m =
          standard_mode_from_spherepack(shift_prime_modes.get(1), l_max, l, m);
      const auto vp_mbar =
          standard_mode_from_spherepack(shift_prime_modes.get(2), l_max, l, m);
      const auto t_ll =
          standard_mode_from_spherepack(metric_modes.get(0, 0), l_max, l, m);
      const auto t_lm =
          standard_mode_from_spherepack(metric_modes.get(0, 1), l_max, l, m);
      const auto t_lmbar =
          standard_mode_from_spherepack(metric_modes.get(0, 2), l_max, l, m);
      const auto t_mm =
          standard_mode_from_spherepack(metric_modes.get(1, 1), l_max, l, m);
      const auto t_mmbar =
          standard_mode_from_spherepack(metric_modes.get(1, 2), l_max, l, m);
      const auto t_mbarmbar =
          standard_mode_from_spherepack(metric_modes.get(2, 2), l_max, l, m);
      const auto dt_lm =
          standard_mode_from_spherepack(dt_metric_modes.get(0, 1), l_max, l, m);
      const auto dt_lmbar =
          standard_mode_from_spherepack(dt_metric_modes.get(0, 2), l_max, l, m);
      const auto dr_mm =
          standard_mode_from_spherepack(dr_metric_modes.get(1, 1), l_max, l, m);
      const auto dr_mmbar =
          standard_mode_from_spherepack(dr_metric_modes.get(1, 2), l_max, l, m);
      const auto dr_mbarmbar =
          standard_mode_from_spherepack(dr_metric_modes.get(2, 2), l_max, l, m);
      h_t[goldberg_index(l_max, l, m)] =
          i * extraction_radius * vector_prefactor * (v_mbar + v_m);
      dr_h_t[goldberg_index(l_max, l, m)] =
          i * vector_prefactor *
          (extraction_radius * (vp_mbar + vp_m) + (v_mbar + v_m));
      dt_h_r[goldberg_index(l_max, l, m)] =
          i * extraction_radius * vector_prefactor * (dt_lmbar + dt_lm);
      h_rr[goldberg_index(l_max, l, m)] = t_ll;
      q_r[goldberg_index(l_max, l, m)] =
          -extraction_radius * vector_prefactor * (t_lmbar - t_lm);
      k[goldberg_index(l_max, l, m)] = t_mmbar;
      dr_k[goldberg_index(l_max, l, m)] = dr_mmbar;
      g[goldberg_index(l_max, l, m)] = tensor_prefactor * (t_mbarmbar + t_mm);
      dr_g[goldberg_index(l_max, l, m)] =
          tensor_prefactor * (dr_mbarmbar + dr_mm);
    }
  }

  regge_wheeler_zerilli_moncrief(rwz_quantities, h_t, dr_h_t, dt_h_r, h_rr, q_r,
                                 k, dr_k, g, dr_g, l_max, extraction_radius);
}

ReggeWheelerZerilli regge_wheeler_zerilli_moncrief_from_gh_vars(
    const tnsr::aa<DataVector, 3, Frame::Inertial>& spacetime_metric,
    const tnsr::aa<DataVector, 3, Frame::Inertial>& pi,
    const tnsr::iaa<DataVector, 3, Frame::Inertial>& phi,
    const tnsr::I<DataVector, 3, Frame::Inertial>& inertial_coords,
    const ylm::Spherepack& ylm_spherepack, const std::array<double, 3>& center,
    const double extraction_radius) {
  ReggeWheelerZerilli rwz_quantities{ylm_spherepack.l_max()};
  regge_wheeler_zerilli_moncrief_from_gh_vars(
      make_not_null(&rwz_quantities), spacetime_metric, pi, phi,
      inertial_coords, ylm_spherepack, center, extraction_radius);
  return rwz_quantities;
}

void psi_4_modes_from_tensors(
    const gsl::not_null<ComplexModalVector*> r_times_psi_4,
    const tnsr::aa<DataVector, 3, Frame::Inertial>& spacetime_metric,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_ricci,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& extrinsic_curvature,
    const tnsr::ijj<DataVector, 3, Frame::Inertial>&
        cov_deriv_extrinsic_curvature,
    const tnsr::I<DataVector, 3, Frame::Inertial>& inertial_coords,
    const ylm::Spherepack& ylm_spherepack, const std::array<double, 3>& center,
    const double extraction_radius) {
  ASSERT(
      extraction_radius > 0.0,
      "The extraction radius must be positive, but is " << extraction_radius);
  ASSERT(get<0>(inertial_coords).size() == ylm_spherepack.physical_size(),
         "Expected one extraction sphere worth of nodal points ("
             << ylm_spherepack.physical_size() << "), but received "
             << get<0>(inertial_coords).size());

  const size_t l_max = ylm_spherepack.l_max();
  const size_t number_of_points = get<0>(inertial_coords).size();

  const auto spatial_metric = gr::spatial_metric(spacetime_metric);
  const auto inverse_spatial_metric =
      determinant_and_inverse(spatial_metric).second;

  tnsr::I<DataVector, 3, Frame::Inertial> centered_coords{number_of_points};
  for (size_t i = 0; i < 3; ++i) {
    centered_coords.get(i) = inertial_coords.get(i) - gsl::at(center, i);
  }

  Scalar<DataVector> centered_radius{number_of_points, 0.0};
  for (size_t s = 0; s < number_of_points; ++s) {
    double radius_squared = 0.0;
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        radius_squared += spatial_metric.get(i, j)[s] *
                          centered_coords.get(i)[s] * centered_coords.get(j)[s];
      }
    }
    get(centered_radius)[s] = sqrt(radius_squared);
  }

  tnsr::I<DataVector, 3, Frame::Inertial> radial_unit_vector{number_of_points};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t s = 0; s < number_of_points; ++s) {
      radial_unit_vector.get(i)[s] =
          get(centered_radius)[s] > 0.0
              ? centered_coords.get(i)[s] / get(centered_radius)[s]
              : 0.0;
    }
  }

  tnsr::i<DataVector, 3, Frame::Inertial> radial_unit_one_form{number_of_points,
                                                               0.0};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      radial_unit_one_form.get(i) +=
          spatial_metric.get(i, j) * radial_unit_vector.get(j);
    }
  }

  const auto projection_tensor =
      gr::transverse_projection_operator(spatial_metric, radial_unit_one_form);
  const auto inverse_projection_tensor = gr::transverse_projection_operator(
      inverse_spatial_metric, radial_unit_vector);
  const auto projection_up_lo = gr::transverse_projection_operator(
      radial_unit_vector, radial_unit_one_form);

  const auto u8_plus = gr::weyl_propagating(
      spatial_ricci, extrinsic_curvature, inverse_spatial_metric,
      cov_deriv_extrinsic_curvature, radial_unit_vector,
      inverse_projection_tensor, projection_tensor, projection_up_lo, 1.0);
  const auto u8_plus_modes =
      cartesian_to_spherical_tensor_modes(u8_plus, ylm_spherepack);

  *r_times_psi_4 = ComplexModalVector{square(l_max + 1), 0.0};
  for (size_t l = 0; l <= l_max; ++l) {
    for (int m = 0; m <= static_cast<int>(l); ++m) {
      set_goldberg_mode(
          r_times_psi_4, l_max, l, m,
          extraction_radius * standard_mode_from_spherepack(
                                  u8_plus_modes.get(1, 1), l_max, l, m));
    }
  }
  fill_negative_m_modes(r_times_psi_4, l_max);
}

ComplexModalVector psi_4_modes_from_tensors(
    const tnsr::aa<DataVector, 3, Frame::Inertial>& spacetime_metric,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_ricci,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& extrinsic_curvature,
    const tnsr::ijj<DataVector, 3, Frame::Inertial>&
        cov_deriv_extrinsic_curvature,
    const tnsr::I<DataVector, 3, Frame::Inertial>& inertial_coords,
    const ylm::Spherepack& ylm_spherepack, const std::array<double, 3>& center,
    const double extraction_radius) {
  ComplexModalVector r_times_psi_4{};
  psi_4_modes_from_tensors(make_not_null(&r_times_psi_4), spacetime_metric,
                           spatial_ricci, extrinsic_curvature,
                           cov_deriv_extrinsic_curvature, inertial_coords,
                           ylm_spherepack, center, extraction_radius);
  return r_times_psi_4;
}

void extraction_sphere_metadata_from_gh_vars(
    const gsl::not_null<ExtractionSphereMetadata*> metadata,
    const tnsr::aa<DataVector, 3, Frame::Inertial>& spacetime_metric,
    const ylm::Strahlkorper<Frame::Inertial>& strahlkorper) {
  const auto spatial_metric = gr::spatial_metric(spacetime_metric);
  const auto inverse_spatial_metric =
      determinant_and_inverse(spatial_metric).second;
  const auto shift = gr::shift(spacetime_metric, inverse_spatial_metric);
  const auto lapse = gr::lapse(shift, spacetime_metric);

  const auto theta_phi = ylm::theta_phi(strahlkorper);
  const auto r_hat = ylm::rhat<Frame::Inertial>(theta_phi);
  const auto jacobian = ylm::jacobian<Frame::Inertial>(theta_phi);
  const auto radius = ylm::radius(strahlkorper);
  tnsr::i<DataVector, 3, Frame::Inertial> zero_dx_radius{get(lapse).size(),
                                                         0.0};
  const auto normal_one_form = ylm::normal_one_form(zero_dx_radius, r_hat);
  const auto proper_area_element = gr::surfaces::area_element(
      spatial_metric, jacobian, normal_one_form, radius, r_hat);

  Scalar<DataVector> unity{get(lapse).size(), 1.0};
  const double proper_area = gr::surfaces::surface_integral_of_scalar(
      proper_area_element, unity, strahlkorper);
  metadata->average_lapse = gr::surfaces::surface_integral_of_scalar(
                                proper_area_element, lapse, strahlkorper) /
                            proper_area;
  metadata->areal_radius = sqrt(proper_area / (4.0 * M_PI));
}

ExtractionSphereMetadata extraction_sphere_metadata_from_gh_vars(
    const tnsr::aa<DataVector, 3, Frame::Inertial>& spacetime_metric,
    const ylm::Strahlkorper<Frame::Inertial>& strahlkorper) {
  ExtractionSphereMetadata metadata{};
  extraction_sphere_metadata_from_gh_vars(make_not_null(&metadata),
                                          spacetime_metric, strahlkorper);
  return metadata;
}

}  // namespace gr::surfaces
