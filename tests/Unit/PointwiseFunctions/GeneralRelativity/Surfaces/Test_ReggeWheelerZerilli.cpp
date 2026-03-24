// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SimpleSparseMatrix.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackIterator.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/StrahlkorperFunctions.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/TensorYlmCartToSphere.hpp"
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

void check_complex_approx(const std::complex<double>& actual,
                          const std::complex<double>& expected) {
  CHECK(real(actual) == approx(real(expected)));
  CHECK(imag(actual) == approx(imag(expected)));
}

double teukolsky_coefficient_a(const double radius, const double time,
                               const double amplitude, const double duration) {
  const double u = time - radius;
  return 3.0 * amplitude * exp(-square(u) / square(duration)) /
         (pow<4>(duration) * pow<5>(radius)) *
         (3.0 * pow<4>(duration) + 4.0 * square(radius) * square(u) -
          2.0 * square(duration) * radius * (radius + 3.0 * u));
}

double teukolsky_coefficient_b(const double radius, const double time,
                               const double amplitude, const double duration) {
  const double u = time - radius;
  return 2.0 * amplitude * exp(-square(u) / square(duration)) /
         (pow<6>(duration) * pow<5>(radius)) *
         (-3.0 * pow<6>(duration) + 4.0 * pow<3>(radius) * pow<3>(u) -
          6.0 * square(duration) * square(radius) * u * (radius + u) +
          3.0 * pow<4>(duration) * radius * (radius + 2.0 * u));
}

double teukolsky_coefficient_c(const double radius, const double time,
                               const double amplitude, const double duration) {
  const double u = time - radius;
  return 0.25 * amplitude * exp(-square(u) / square(duration)) /
         (pow<8>(duration) * pow<5>(radius)) *
         (21.0 * pow<8>(duration) + 16.0 * pow<4>(radius) * pow<4>(u) -
          16.0 * square(duration) * pow<3>(radius) * square(u) *
              (3.0 * radius + u) -
          6.0 * pow<6>(duration) * radius * (3.0 * radius + 7.0 * u) +
          12.0 * pow<4>(duration) * square(radius) *
              (square(radius) + 2.0 * radius * u + 3.0 * square(u)));
}

double dt_teukolsky_coefficient_a(const double radius, const double time,
                                  const double amplitude,
                                  const double duration) {
  const double u = time - radius;
  return -2.0 * u / square(duration) *
             teukolsky_coefficient_a(radius, time, amplitude, duration) +
         3.0 * amplitude * exp(-square(u) / square(duration)) /
             (pow<4>(duration) * pow<5>(radius)) *
             (8.0 * square(radius) * u - 6.0 * square(duration) * radius);
}

double dt_teukolsky_coefficient_b(const double radius, const double time,
                                  const double amplitude,
                                  const double duration) {
  const double u = time - radius;
  return -2.0 * u / square(duration) *
             teukolsky_coefficient_b(radius, time, amplitude, duration) +
         2.0 * amplitude * exp(-square(u) / square(duration)) /
             (pow<6>(duration) * pow<5>(radius)) *
             (12.0 * pow<3>(radius) * square(u) -
              6.0 * square(duration) * square(radius) * (radius + 2.0 * u) +
              6.0 * pow<4>(duration) * radius);
}

double dt_teukolsky_coefficient_c(const double radius, const double time,
                                  const double amplitude,
                                  const double duration) {
  const double u = time - radius;
  return -2.0 * u / square(duration) *
             teukolsky_coefficient_c(radius, time, amplitude, duration) +
         0.25 * amplitude * exp(-square(u) / square(duration)) /
             (pow<8>(duration) * pow<5>(radius)) *
             (64.0 * pow<4>(radius) * pow<3>(u) -
              16.0 * square(duration) * pow<3>(radius) * u *
                  (6.0 * radius + 3.0 * u) -
              42.0 * pow<6>(duration) * radius +
              12.0 * pow<4>(duration) * square(radius) *
                  (2.0 * radius + 6.0 * u));
}

double dr_teukolsky_coefficient_a(const double radius, const double time,
                                  const double amplitude,
                                  const double duration) {
  const double u = time - radius;
  return -dt_teukolsky_coefficient_a(radius, time, amplitude, duration) -
         9.0 * amplitude * exp(-square(u) / square(duration)) /
             (pow<4>(duration) * pow<6>(radius)) *
             (5.0 * pow<4>(duration) + 4.0 * square(radius) * square(u) -
              2.0 * square(duration) * radius * (radius + 4.0 * u));
}

double dr_teukolsky_coefficient_b(const double radius, const double time,
                                  const double amplitude,
                                  const double duration) {
  const double u = time - radius;
  return -dt_teukolsky_coefficient_b(radius, time, amplitude, duration) +
         2.0 * amplitude * exp(-square(u) / square(duration)) /
             (pow<6>(duration) * pow<6>(radius)) *
             (15.0 * pow<6>(duration) - 8.0 * pow<3>(radius) * pow<3>(u) +
              6.0 * square(duration) * square(radius) * u *
                  (2.0 * radius + 3.0 * u) -
              3.0 * pow<4>(duration) * radius * (3.0 * radius + 8.0 * u));
}

double dr_teukolsky_coefficient_c(const double radius, const double time,
                                  const double amplitude,
                                  const double duration) {
  const double u = time - radius;
  return -dt_teukolsky_coefficient_c(radius, time, amplitude, duration) -
         0.25 * amplitude * exp(-square(u) / square(duration)) /
             (pow<8>(duration) * pow<6>(radius)) *
             (105.0 * pow<8>(duration) + 16.0 * pow<4>(radius * u) -
              16.0 * square(duration) * pow<3>(radius) * square(u) *
                  (3.0 * radius + 2.0 * u) -
              6.0 * pow<6>(duration) * radius * (9.0 * radius + 28.0 * u) +
              12.0 * pow<4>(duration) * square(radius) *
                  (square(radius) + 4.0 * radius * u + 9.0 * square(u)));
}

tnsr::Ij<DataVector, 3, Frame::Inertial> teukolsky_inverse_jacobian(
    const tnsr::i<DataVector, 2, Frame::Spherical<Frame::Inertial>>& theta_phi,
    const double radius) {
  const size_t number_of_points = get<0>(theta_phi).size();
  tnsr::Ij<DataVector, 3, Frame::Inertial> inverse_jacobian{number_of_points,
                                                            0.0};
  for (size_t s = 0; s < number_of_points; ++s) {
    const double theta = theta_phi.get(0)[s];
    const double phi = theta_phi.get(1)[s];
    inverse_jacobian.get(0, 0)[s] = cos(phi) * sin(theta);
    inverse_jacobian.get(0, 1)[s] = cos(phi) * cos(theta) / radius;
    inverse_jacobian.get(0, 2)[s] = -sin(phi) / radius;
    inverse_jacobian.get(1, 0)[s] = sin(phi) * sin(theta);
    inverse_jacobian.get(1, 1)[s] = cos(theta) * sin(phi) / radius;
    inverse_jacobian.get(1, 2)[s] = cos(phi) / radius;
    inverse_jacobian.get(2, 0)[s] = cos(theta);
    inverse_jacobian.get(2, 1)[s] = -sin(theta) / radius;
    inverse_jacobian.get(2, 2)[s] = 0.0;
  }
  return inverse_jacobian;
}

tnsr::Ij<DataVector, 3, Frame::Inertial> teukolsky_dr_inverse_jacobian(
    const tnsr::i<DataVector, 2, Frame::Spherical<Frame::Inertial>>& theta_phi,
    const double radius) {
  const size_t number_of_points = get<0>(theta_phi).size();
  tnsr::Ij<DataVector, 3, Frame::Inertial> dr_inverse_jacobian{number_of_points,
                                                               0.0};
  for (size_t s = 0; s < number_of_points; ++s) {
    const double theta = theta_phi.get(0)[s];
    const double phi = theta_phi.get(1)[s];
    dr_inverse_jacobian.get(0, 0)[s] = 0.0;
    dr_inverse_jacobian.get(0, 1)[s] = -cos(phi) * cos(theta) / square(radius);
    dr_inverse_jacobian.get(0, 2)[s] = sin(phi) / square(radius);
    dr_inverse_jacobian.get(1, 0)[s] = 0.0;
    dr_inverse_jacobian.get(1, 1)[s] = -cos(theta) * sin(phi) / square(radius);
    dr_inverse_jacobian.get(1, 2)[s] = -cos(phi) / square(radius);
    dr_inverse_jacobian.get(2, 0)[s] = 0.0;
    dr_inverse_jacobian.get(2, 1)[s] = sin(theta) / square(radius);
    dr_inverse_jacobian.get(2, 2)[s] = 0.0;
  }
  return dr_inverse_jacobian;
}

template <typename TensorType>
TensorType transform_spherical_to_cartesian(
    const TensorType& spherical_tensor,
    const tnsr::Ij<DataVector, 3, Frame::Inertial>& inverse_jacobian) {
  TensorType cartesian_tensor{get<0, 0>(spherical_tensor).size(), 0.0};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      for (size_t a = 0; a < 3; ++a) {
        for (size_t b = 0; b < 3; ++b) {
          cartesian_tensor.get(i, j) += inverse_jacobian.get(i, a) *
                                        inverse_jacobian.get(j, b) *
                                        spherical_tensor.get(a, b);
        }
      }
    }
  }
  return cartesian_tensor;
}

tnsr::ii<DataVector, 3, Frame::Inertial> transform_dr_spherical_to_cartesian(
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spherical_tensor,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& dr_spherical_tensor,
    const tnsr::Ij<DataVector, 3, Frame::Inertial>& inverse_jacobian,
    const tnsr::Ij<DataVector, 3, Frame::Inertial>& dr_inverse_jacobian) {
  tnsr::ii<DataVector, 3, Frame::Inertial> dr_cartesian{
      get<0, 0>(spherical_tensor).size(), 0.0};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      for (size_t a = 0; a < 3; ++a) {
        for (size_t b = 0; b < 3; ++b) {
          dr_cartesian.get(i, j) +=
              inverse_jacobian.get(i, a) * inverse_jacobian.get(j, b) *
                  dr_spherical_tensor.get(a, b) +
              (dr_inverse_jacobian.get(i, a) * inverse_jacobian.get(j, b) +
               inverse_jacobian.get(i, a) * dr_inverse_jacobian.get(j, b)) *
                  spherical_tensor.get(a, b);
        }
      }
    }
  }
  return dr_cartesian;
}

gr::surfaces::ReggeWheelerZerilli teukolsky_rwz_reference(
    const size_t l_max, const double radius, const double time,
    const double amplitude, const double duration) {
  const ylm::Strahlkorper<Frame::Inertial> strahlkorper{
      l_max, l_max, radius, {{0.0, 0.0, 0.0}}};
  const auto theta_phi = ylm::theta_phi(strahlkorper);
  const size_t number_of_points = get<0>(theta_phi).size();
  const auto inverse_jacobian = teukolsky_inverse_jacobian(theta_phi, radius);
  const auto dr_inverse_jacobian =
      teukolsky_dr_inverse_jacobian(theta_phi, radius);

  const double coefficient_a =
      teukolsky_coefficient_a(radius, time, amplitude, duration);
  const double coefficient_b =
      teukolsky_coefficient_b(radius, time, amplitude, duration);
  const double coefficient_c =
      teukolsky_coefficient_c(radius, time, amplitude, duration);
  const double dt_coefficient_a =
      dt_teukolsky_coefficient_a(radius, time, amplitude, duration);
  const double dt_coefficient_b =
      dt_teukolsky_coefficient_b(radius, time, amplitude, duration);
  const double dt_coefficient_c =
      dt_teukolsky_coefficient_c(radius, time, amplitude, duration);
  const double dr_coefficient_a =
      dr_teukolsky_coefficient_a(radius, time, amplitude, duration);
  const double dr_coefficient_b =
      dr_teukolsky_coefficient_b(radius, time, amplitude, duration);
  const double dr_coefficient_c =
      dr_teukolsky_coefficient_c(radius, time, amplitude, duration);

  tnsr::ii<DataVector, 3, Frame::Inertial> spherical_metric_perturbation{
      number_of_points, 0.0};
  tnsr::ii<DataVector, 3, Frame::Inertial> dt_spherical_metric_perturbation{
      number_of_points, 0.0};
  tnsr::ii<DataVector, 3, Frame::Inertial> dr_spherical_metric_perturbation{
      number_of_points, 0.0};

  for (size_t s = 0; s < number_of_points; ++s) {
    const double theta = theta_phi.get(0)[s];
    const double sin_theta = sin(theta);
    const double cos_theta = cos(theta);
    const double sin_sq = square(sin_theta);
    const double f_rr = 2.0 - 3.0 * sin_sq;
    const double f_r_theta = -3.0 * sin_theta * cos_theta;
    const double f_c_theta_theta = 3.0 * sin_sq;
    const double f_a_theta_theta = -1.0;
    const double f_c_phi_phi = -3.0 * sin_sq;
    const double f_a_phi_phi = 3.0 * sin_sq - 1.0;

    spherical_metric_perturbation.get(0, 0)[s] = coefficient_a * f_rr;
    spherical_metric_perturbation.get(0, 1)[s] =
        coefficient_b * f_r_theta * radius;
    spherical_metric_perturbation.get(1, 1)[s] =
        (coefficient_c * f_c_theta_theta + coefficient_a * f_a_theta_theta) *
        square(radius);
    spherical_metric_perturbation.get(2, 2)[s] =
        (coefficient_c * f_c_phi_phi + coefficient_a * f_a_phi_phi) *
        square(radius);

    dt_spherical_metric_perturbation.get(0, 0)[s] = dt_coefficient_a * f_rr;
    dt_spherical_metric_perturbation.get(0, 1)[s] =
        dt_coefficient_b * f_r_theta * radius;
    dt_spherical_metric_perturbation.get(1, 1)[s] =
        (dt_coefficient_c * f_c_theta_theta +
         dt_coefficient_a * f_a_theta_theta) *
        square(radius);
    dt_spherical_metric_perturbation.get(2, 2)[s] =
        (dt_coefficient_c * f_c_phi_phi + dt_coefficient_a * f_a_phi_phi) *
        square(radius);

    dr_spherical_metric_perturbation.get(0, 0)[s] = dr_coefficient_a * f_rr;
    dr_spherical_metric_perturbation.get(0, 1)[s] =
        (coefficient_b + radius * dr_coefficient_b) * f_r_theta;
    dr_spherical_metric_perturbation.get(1, 1)[s] =
        radius *
        (2.0 +
         (2.0 * coefficient_c + radius * dr_coefficient_c) * f_c_theta_theta +
         (2.0 * coefficient_a + radius * dr_coefficient_a) * f_a_theta_theta);
    dr_spherical_metric_perturbation.get(2, 2)[s] =
        radius *
        (2.0 + (2.0 * coefficient_c + radius * dr_coefficient_c) * f_c_phi_phi +
         (2.0 * coefficient_a + radius * dr_coefficient_a) * f_a_phi_phi);
  }

  const auto spatial_metric_perturbation = transform_spherical_to_cartesian(
      spherical_metric_perturbation, inverse_jacobian);
  const auto dt_spatial_metric = transform_spherical_to_cartesian(
      dt_spherical_metric_perturbation, inverse_jacobian);
  const auto dr_spatial_metric = transform_dr_spherical_to_cartesian(
      spherical_metric_perturbation, dr_spherical_metric_perturbation,
      inverse_jacobian, dr_inverse_jacobian);

  const auto metric_modes = cartesian_to_spherical_tensor_modes(
      spatial_metric_perturbation, strahlkorper.ylm_spherepack());
  const auto dt_metric_modes = cartesian_to_spherical_tensor_modes(
      dt_spatial_metric, strahlkorper.ylm_spherepack());
  const auto dr_metric_modes = cartesian_to_spherical_tensor_modes(
      dr_spatial_metric, strahlkorper.ylm_spherepack());

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

  for (size_t l = 2; l <= l_max; ++l) {
    const double vector_prefactor =
        1.0 / sqrt(2.0 * static_cast<double>(l * (l + 1)));
    const double tensor_prefactor =
        1.0 / sqrt(static_cast<double>((l - 1) * l * (l + 1) * (l + 2)));
    for (int m = 0; m <= static_cast<int>(l); ++m) {
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
      h_rr[goldberg_index(l_max, l, m)] = t_ll;
      q_r[goldberg_index(l_max, l, m)] =
          -radius * vector_prefactor * (t_lmbar - t_lm);
      k[goldberg_index(l_max, l, m)] = t_mmbar;
      dr_k[goldberg_index(l_max, l, m)] = dr_mmbar;
      g[goldberg_index(l_max, l, m)] = tensor_prefactor * (t_mbarmbar + t_mm);
      dr_g[goldberg_index(l_max, l, m)] =
          tensor_prefactor * (dr_mbarmbar + dr_mm);
      dt_h_r[goldberg_index(l_max, l, m)] = std::complex<double>{0.0, 1.0} *
                                            radius * vector_prefactor *
                                            (dt_lmbar + dt_lm);
    }
  }

  return gr::surfaces::regge_wheeler_zerilli_moncrief(
      h_t, dr_h_t, dt_h_r, h_rr, q_r, k, dr_k, g, dr_g, l_max, radius);
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

void test_regge_wheeler_zerilli_teukolsky_reference_mode_structure() {
  const size_t l_max = 6;
  const double radius = 10.0;
  const double duration = 1.4;
  const double time = radius + 0.6 * duration;
  const double amplitude = 0.03;

  const auto rwz =
      teukolsky_rwz_reference(l_max, radius, time, amplitude, duration);
  const auto rwz_double_amplitude =
      teukolsky_rwz_reference(l_max, radius, time, 2.0 * amplitude, duration);

  const auto phi_plus_20 = mode(rwz.phi_plus, l_max, 2, 0);
  const auto phi_minus_20 = mode(rwz.phi_minus, l_max, 2, 0);
  const auto strain_20 = mode(rwz.r_times_strain, l_max, 2, 0);

  CAPTURE(phi_plus_20);
  CAPTURE(phi_minus_20);
  CAPTURE(strain_20);

  CHECK(abs(phi_plus_20) > 1.0e-4);
  CHECK(abs(strain_20) > 1.0e-4);
  CHECK(abs(phi_minus_20) < 1.0e-10);

  check_complex_approx(mode(rwz_double_amplitude.phi_plus, l_max, 2, 0),
                       2.0 * phi_plus_20);
  check_complex_approx(mode(rwz_double_amplitude.phi_minus, l_max, 2, 0),
                       2.0 * phi_minus_20);
  check_complex_approx(mode(rwz_double_amplitude.r_times_strain, l_max, 2, 0),
                       2.0 * strain_20);

  for (size_t l = 2; l <= l_max; ++l) {
    for (int m = -static_cast<int>(l); m <= static_cast<int>(l); ++m) {
      if (l == 2 and m == 0) {
        continue;
      }
      CAPTURE(l);
      CAPTURE(m);
      CHECK(abs(mode(rwz.phi_plus, l_max, l, m)) < 1.0e-10);
      CHECK(abs(mode(rwz.phi_minus, l_max, l, m)) < 1.0e-10);
      CHECK(abs(mode(rwz.r_times_strain, l_max, l, m)) < 1.0e-10);
    }
  }
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
  test_regge_wheeler_zerilli_teukolsky_reference_mode_structure();
  test_regge_wheeler_zerilli_from_gh_vars_minkowski();
  test_regge_wheeler_zerilli_from_gh_vars_kerr_schild_schwarzschild();
  test_extraction_sphere_metadata_from_gh_vars_minkowski();
  test_psi_4_modes_from_tensors_minkowski();
}
