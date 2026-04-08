// Distributed under the MIT License.
// See LICENSE.txt for details.

// \file
// Tests of power monitors.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/PowerMonitors.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/BasisFunctionValue.hpp"
#include "NumericalAlgorithms/Spectral/CollocationPoints.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackIterator.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/TensorYlmHelpers.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/TensorYlmTransforms.hpp"
#include "Utilities/Array.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace {

void test_power_monitors_impl() {
  const size_t number_of_points_per_dimension = 4;
  const size_t number_of_points = pow<2>(number_of_points_per_dimension);

  // Test a constant function
  const DataVector test_data_vector{number_of_points, 1.0};
  const ComplexDataVector test_complex_data_vector{
      number_of_points, std::complex<double>(1.0, 1.0)};

  const Mesh<2_st> mesh{number_of_points_per_dimension,
                        Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};

  const auto test_power_monitors =
      PowerMonitors::power_monitors(test_data_vector, mesh);
  const auto test_power_monitors_complex =
      PowerMonitors::power_monitors(test_complex_data_vector, mesh);

  // The only non-zero modal coefficient of a constant is the one corresponding
  // to the first Legendre polynomial
  DataVector check_data_vector =
      DataVector{number_of_points_per_dimension, 0.0};
  check_data_vector[0] = 1.0 / sqrt(number_of_points_per_dimension);
  DataVector check_data_vector_complex =
      DataVector{number_of_points_per_dimension, 0.0};
  check_data_vector_complex[0] = sqrt(2) / sqrt(number_of_points_per_dimension);

  const std::array<DataVector, 2> expected_power_monitors{check_data_vector,
                                                          check_data_vector};
  const std::array<DataVector, 2> expected_power_monitors_complex{
      check_data_vector_complex, check_data_vector_complex};

  CHECK_ITERABLE_APPROX(test_power_monitors, expected_power_monitors);
  CHECK_ITERABLE_APPROX(test_power_monitors_complex,
                        expected_power_monitors_complex);
}

void test_power_monitors_second_impl() {
  const size_t number_of_points_per_dimension = 4;

  const Mesh<2_st> mesh{number_of_points_per_dimension,
                        Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};

  const auto logical_coords = logical_coordinates(mesh);

  // Build a test function containing only one Legendre basis function
  // per dimension
  const size_t x_mode = 0;
  const size_t y_mode = 1;
  const std::array<size_t, 2> coeff = {x_mode, y_mode};

  DataVector u_nodal(mesh.number_of_grid_points(), 1.0);
  for (size_t dim = 0; dim < 2; ++dim) {
    u_nodal *=
        Spectral::compute_basis_function_value<Spectral::Basis::Legendre>(
            gsl::at(coeff, dim), logical_coords.get(dim));
  }

  const auto test_power_monitors = PowerMonitors::power_monitors(u_nodal, mesh);

  // The only non-zero modal coefficient of a constant is the one corresponding
  // to the specified Legendre polynomial

  // In the x direction
  DataVector check_data_vector_x =
      DataVector{number_of_points_per_dimension, 0.0};
  check_data_vector_x[x_mode] = 1.0 / sqrt(number_of_points_per_dimension);

  // In the y direction
  DataVector check_data_vector_y =
      DataVector{number_of_points_per_dimension, 0.0};
  check_data_vector_y[y_mode] = 1.0 / sqrt(number_of_points_per_dimension);

  // We compare against the expected array
  const std::array<DataVector, 2> expected_power_monitors{check_data_vector_x,
                                                          check_data_vector_y};

  CHECK_ITERABLE_APPROX(test_power_monitors, expected_power_monitors);
}

void test_relative_truncation_error_impl() {
  // We recompute the truncation error for a function where we know the
  // power monitors analytically
  const size_t number_of_points_per_dimension = 8;
  const Mesh<1_st> mesh{number_of_points_per_dimension,
                        Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  const auto logical_coords = logical_coordinates(mesh);

  // Build a test function with no zero power monitors
  const std::vector<int> coeffs = {0, 1, 2, 3, 4, 5, 6, 7};
  DataVector u_nodal(mesh.number_of_grid_points(), 0.0);
  double ampl = 0.0;
  for (auto coeff : coeffs) {
    ampl = pow(10.0, -coeff);
    u_nodal +=
        ampl *
        Spectral::compute_basis_function_value<Spectral::Basis::Legendre>(
            static_cast<size_t>(coeff), logical_coords.get(0_st));
  }

  // Compute the relative truncation error
  const int last_coeff = 7;
  double weight = 0.0;
  double avg = 0.0;
  double weight_sum = 0.0;
  for (auto coeff : coeffs) {
    ampl = pow(10.0, -coeff);
    weight = exp(-square(coeff - last_coeff + 0.5));
    avg += log10(ampl) * weight;
    weight_sum += weight;
  }
  avg = avg / weight_sum;
  // By construction the maximum of the magnitude of the first two modes is
  // unity.
  // We test the order of magnitude of the relative error
  const double expected_relative_truncation_error = pow(10.0, avg);

  const auto power_monitors = PowerMonitors::power_monitors(u_nodal, mesh);
  const DataVector& power_monitor_x = gsl::at(power_monitors, 0_st);
  // We use all of the modes as above
  const double test_relative_truncation_error =
      PowerMonitors::relative_truncation_error(power_monitor_x,
                                               power_monitor_x.size());

  CHECK_ITERABLE_APPROX(expected_relative_truncation_error,
                        test_relative_truncation_error);

  // Test truncation error
  const double test_truncation_error =
      PowerMonitors::absolute_truncation_error(u_nodal, mesh)[0];

  // Compare with the result from the relative truncation error
  const double expected_truncation_error_x =
      max(abs(u_nodal)) * PowerMonitors::relative_truncation_error(
                              power_monitor_x, power_monitor_x.size());

  CHECK_ITERABLE_APPROX(test_truncation_error, expected_truncation_error_x);
}

void test_relative_truncation_error_with_symmetry() {
  // Try to resolve half a period of a sinusoid
  const size_t num_modes = 12;
  const Mesh<1> mesh{num_modes, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
  const auto xi = Spectral::collocation_points(mesh);
  const double wave_number = 0.5;
  const DataVector u_nodal = sin((xi + 1.) * M_PI * wave_number);
  CAPTURE(u_nodal);
  auto modes = PowerMonitors::power_monitors(u_nodal, mesh)[0];
  // Add some more noise to the modes
  modes += 10. * std::numeric_limits<double>::epsilon();
  CAPTURE(modes);
  const double relative_truncation_error =
      PowerMonitors::relative_truncation_error(modes, num_modes);
  // Last mode should be zero by symmetry
  REQUIRE(modes[num_modes - 1] == approx(0.));
  // Expect the relative truncation error to be the ratio of the first and last
  // nonzero modes
  const double expected_relative_truncation_error =
      modes[num_modes - 2] / modes[0];
  const Approx custom_approx = Approx::custom().epsilon(5e-2);
  CHECK(relative_truncation_error ==
        custom_approx(expected_relative_truncation_error));
}

void test_relative_truncation_error_linear_function() {
  // Resolve a linear function with a few modes. We technically need only 2.
  const auto get_modes = [](const size_t num_modes) {
    const Mesh<1> mesh{num_modes, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
    const auto xi = Spectral::collocation_points(mesh);
    const DataVector u_nodal = (xi + 1.) * 0.5;
    auto modes = PowerMonitors::power_monitors(u_nodal, mesh)[0];
    // Add some noise to the modes
    modes += 10. * std::numeric_limits<double>::epsilon();
    const double relative_truncation_error =
        PowerMonitors::relative_truncation_error(modes, num_modes);
    return std::make_pair(modes, relative_truncation_error);
  };
  {
    INFO("2 modes");
    const auto [modes, rel_error] = get_modes(2);
    CAPTURE(modes);
    CHECK_ITERABLE_APPROX(modes, (DataVector{0.5, 0.5}));
    // We don't know for sure that we have resolved the function exactly,
    // because we have two nonzero modes and nothing else.
    CHECK(rel_error == approx(1.));
  }
  {
    INFO("3 modes");
    const auto [modes, rel_error] = get_modes(3);
    CAPTURE(modes);
    CHECK_ITERABLE_APPROX(modes, (DataVector{0.5, 0.5, 0.}));
    // The last mode is zero, but we still don't know if we have resolved the
    // function because the last mode could be zero by symmetry.
    CHECK(rel_error == approx(1.));
  }
  {
    INFO("4 modes");
    const auto [modes, rel_error] = get_modes(4);
    CAPTURE(modes);
    CHECK_ITERABLE_APPROX(modes, (DataVector{0.5, 0.5, 0., 0.}));
    // We have two zero modes, so we know we have resolved the function exactly.
    CHECK(rel_error < 1.e-14);
  }
}

void test_convergence_rate() {
  // First, check that a power monitor with an exact, constant slope has the
  // expected convergence rate
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> slope_dis(-4.0, -1.0);
  const double expected_slope = slope_dis(gen);
  std::uniform_real_distribution<> offset_dis(-0.4, -0.1);
  const double offset = offset_dis(gen);
  const size_t size_of_power_monitor{10};
  DataVector power_monitor_with_known_slope{size_of_power_monitor};
  for (size_t i = 0; i < size_of_power_monitor; ++i) {
    power_monitor_with_known_slope[i] =
        pow(10.0, static_cast<double>(i) * expected_slope + offset);
  }
  constexpr size_t filtered_modes = 2;

  double convergence_rate =
      PowerMonitors::convergence_rate_and_number_of_pile_up_modes(
          power_monitor_with_known_slope, filtered_modes)
          .convergence_rate;
  CHECK(approx(convergence_rate) == -expected_slope);

  // Change the filtered modes' power to a NaN, and ensure that this mode
  // is ignored when computing the convergence rate.
  power_monitor_with_known_slope[8] =
      std::numeric_limits<double>::signaling_NaN();
  power_monitor_with_known_slope[9] =
      std::numeric_limits<double>::signaling_NaN();
  convergence_rate =
      PowerMonitors::convergence_rate_and_number_of_pile_up_modes(
          power_monitor_with_known_slope, filtered_modes)
          .convergence_rate;
  CHECK(approx(convergence_rate) == -expected_slope);

  // Test that adding noise of amplitude 1e-2 affects the slope recovered
  // by no more than that amount
  constexpr double noise_amp = 0.01;
  std::uniform_real_distribution<> noise_dis(-noise_amp, noise_amp);
  for (size_t i = 0; i < size_of_power_monitor - filtered_modes; ++i) {
    power_monitor_with_known_slope[i] *= pow(10.0, noise_dis(gen));
  }
  convergence_rate =
      PowerMonitors::convergence_rate_and_number_of_pile_up_modes(
          power_monitor_with_known_slope, filtered_modes)
          .convergence_rate;
  // define custom approx for higher derivative checks
  const Approx custom_approx = Approx::custom().epsilon(noise_amp).scale(1.0);
  CHECK(custom_approx(convergence_rate) == -expected_slope);

// Check assert that sufficient modes were provided
#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(
      PowerMonitors::convergence_rate_and_number_of_pile_up_modes(
          power_monitor_with_known_slope, size_of_power_monitor - 3),
      Catch::Matchers::ContainsSubstring(
          "Power monitor needs at least 4 unfiltered modes to compute "
          "convergence"));
#endif
}

void test_pile_up_modes() {
  // Check assert that sufficient modes were provided
  // First, check that a power monitor with an exact, constant slope has the
  // vanishing pile up modes
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> slope_dis(-2.0, -1.0);
  const double expected_slope = slope_dis(gen);
  std::uniform_real_distribution<> offset_dis(-0.4, -0.1);
  const double offset = offset_dis(gen);
  const size_t size_of_power_monitor{20};
  DataVector power_monitor_with_known_slope{size_of_power_monitor};
  for (size_t i = 0; i < size_of_power_monitor; ++i) {
    power_monitor_with_known_slope[i] =
        pow(10.0, static_cast<double>(i) * expected_slope + offset);
  }
  constexpr size_t filtered_modes = 2;

  const double pile_up_modes_known_slope =
      PowerMonitors::convergence_rate_and_number_of_pile_up_modes(
          power_monitor_with_known_slope, filtered_modes)
          .number_of_pile_up_modes;
  const Approx custom_approx = Approx::custom().epsilon(1.e-10).scale(1.0);
  CHECK(custom_approx(pile_up_modes_known_slope) == 0.0);

  // Revise the power monitor to artificially introduce pile up modes
  // Set the top n modes to be equal to the n-1 power, so the slope is zero.
  // Because the number of pile up modes is defined as a double, the computed
  // pile up mode count will have a fractional part as well as the expected
  // integer number of pile up modes. In the test, ignore the fractional part,
  // but make sure that the expected integer number of pile up modes is
  // recovered. Always leave at least 2 unfiltered modes not piled up.
  for (size_t expected_pile_up_modes = 1;
       expected_pile_up_modes < size_of_power_monitor - filtered_modes - 2;
       ++expected_pile_up_modes) {
    DataVector power_monitor_with_pile_up_modes =
        power_monitor_with_known_slope;
    for (size_t i =
             size_of_power_monitor - filtered_modes - expected_pile_up_modes;
         i < size_of_power_monitor - filtered_modes; ++i) {
      power_monitor_with_pile_up_modes[i] =
          power_monitor_with_pile_up_modes[size_of_power_monitor -
                                           filtered_modes -
                                           expected_pile_up_modes - 1];
    }
    // Ensure that filtered modes are not used by replacing them with NaN
    for (size_t i = size_of_power_monitor - filtered_modes;
         i < size_of_power_monitor; ++i) {
      power_monitor_with_pile_up_modes[i] =
          std::numeric_limits<double>::signaling_NaN();
    }
    const double pile_up_modes =
        PowerMonitors::convergence_rate_and_number_of_pile_up_modes(
            power_monitor_with_pile_up_modes, filtered_modes)
            .number_of_pile_up_modes;
    CHECK(static_cast<size_t>(std::floor(pile_up_modes)) ==
          expected_pile_up_modes);
  }

  // Check that a power monitor with zero convergence rate returns zero piled up
  // modes
  power_monitor_with_known_slope = power_monitor_with_known_slope[0];
  const double pile_up_modes_zero_convergence =
      PowerMonitors::convergence_rate_and_number_of_pile_up_modes(
          power_monitor_with_known_slope, filtered_modes)
          .number_of_pile_up_modes;
  CHECK(pile_up_modes_zero_convergence == 0.0);
}

void test_shell_power_monitors() {
  const Mesh<3> shell_mesh{
      {3_st, 4_st, 7_st},
      {Spectral::Basis::Legendre, Spectral::Basis::SphericalHarmonic,
       Spectral::Basis::SphericalHarmonic},
      {Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::Equiangular,
       Spectral::Quadrature::Equiangular}};

  const DataVector scalar_data(shell_mesh.number_of_grid_points(), 1.0);
  const auto scalar_buffer =
      PowerMonitors::shell_power_monitor_buffer(scalar_data, shell_mesh);
  const auto scalar_monitors =
      PowerMonitors::shell_power_monitors(scalar_data, shell_mesh);
  const auto finalized_scalar =
      PowerMonitors::finalize_shell_power_monitor_buffer(scalar_buffer);

  CHECK_ITERABLE_APPROX(scalar_monitors.radial, finalized_scalar.radial);
  CHECK_ITERABLE_APPROX(scalar_monitors.angular, finalized_scalar.angular);
  CHECK(scalar_monitors.radial[0] > 0.0);
  CHECK(scalar_monitors.angular[0] > 0.0);
  for (size_t i = 1; i < scalar_monitors.radial.size(); ++i) {
    CHECK(scalar_monitors.radial[i] == approx(0.0));
  }
  for (size_t i = 1; i < scalar_monitors.angular.size(); ++i) {
    CHECK(scalar_monitors.angular[i] == approx(0.0));
  }

  tnsr::i<DataVector, 3> vector_data(shell_mesh.number_of_grid_points(), 0.0);
  vector_data.get(0) = DataVector(shell_mesh.number_of_grid_points(), 1.0);
  const auto tensor_buffer =
      PowerMonitors::shell_power_monitor_buffer(vector_data, shell_mesh);
  const auto tensor_monitors =
      PowerMonitors::shell_power_monitors(vector_data, shell_mesh);
  const auto finalized_tensor =
      PowerMonitors::finalize_shell_power_monitor_buffer(tensor_buffer);
  const Scalar<DataVector> scalar_tensor{scalar_data};
  const auto scalar_tensor_buffer =
      PowerMonitors::shell_power_monitor_buffer(scalar_tensor, shell_mesh);
  const auto scalar_tensor_monitors =
      PowerMonitors::shell_power_monitors(scalar_tensor, shell_mesh);

  CHECK_ITERABLE_APPROX(tensor_monitors.radial, finalized_tensor.radial);
  CHECK_ITERABLE_APPROX(tensor_monitors.angular, finalized_tensor.angular);
  CHECK(tensor_monitors.radial[0] > 0.0);
  CHECK_ITERABLE_APPROX(scalar_tensor_monitors.radial, scalar_monitors.radial);
  CHECK_ITERABLE_APPROX(scalar_tensor_monitors.angular,
                        scalar_monitors.angular);
  CHECK(scalar_tensor_buffer.angular_counts == scalar_buffer.angular_counts);
  CHECK_ITERABLE_APPROX(
      PowerMonitors::finalize_shell_power_monitor_buffer(scalar_tensor_buffer)
          .angular,
      scalar_monitors.angular);

  const auto call_shell_power_on_non_shell_mesh = []() {
    const Mesh<3> mesh{3_st, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
    const DataVector data(mesh.number_of_grid_points(), 1.0);
    (void)PowerMonitors::shell_power_monitors(data, mesh);
  };
  CHECK_THROWS_WITH(
      call_shell_power_on_non_shell_mesh(),
      Catch::Matchers::ContainsSubstring(
          "Shell power monitors support exactly one non-angular dimension"));

  const auto call_shell_power_on_non_radial_first_shell_mesh = []() {
    const Mesh<3> mesh{
        {4_st, 7_st, 3_st},
        {Spectral::Basis::SphericalHarmonic, Spectral::Basis::SphericalHarmonic,
         Spectral::Basis::Legendre},
        {Spectral::Quadrature::Gauss, Spectral::Quadrature::Equiangular,
         Spectral::Quadrature::GaussLobatto}};
    const DataVector data(mesh.number_of_grid_points(), 1.0);
    (void)PowerMonitors::shell_power_monitors(data, mesh);
  };
  CHECK_THROWS_WITH(
      call_shell_power_on_non_radial_first_shell_mesh(),
      Catch::Matchers::ContainsSubstring(
          "radial dimension first and the two spherical-harmonic dimensions "
          "last"));
}

std::vector<size_t> scalar_tensor_angular_counts(const size_t n_r,
                                                 const size_t l_max) {
  std::vector<size_t> result(l_max + 1, 0);
  for (size_t l = 0; l <= l_max; ++l) {
    result[l] = n_r * (2 * l + 1);
  }
  return result;
}

std::vector<size_t> vector_tensor_angular_counts(const size_t n_r,
                                                 const size_t l_max) {
  std::vector<size_t> result(l_max + 1, 0);
  if (l_max + 1 > 0) {
    result[0] = 3 * n_r;
  }
  for (size_t l = 1; l <= l_max; ++l) {
    result[l] = n_r * 3 * 2 * (l + 1);
  }
  return result;
}

std::vector<size_t> symmetric_rank2_tensor_angular_counts(const size_t n_r,
                                                          const size_t l_max) {
  std::vector<size_t> result(l_max + 1, 0);
  if (l_max + 1 > 0) {
    result[0] = 6 * n_r;
  }
  if (l_max + 1 > 1) {
    result[1] = 16 * n_r;
  }
  for (size_t l = 2; l <= l_max; ++l) {
    result[l] = n_r * 6 * 2 * (l + 1);
  }
  return result;
}

void test_shell_tensor_count_semantics() {
  const Mesh<3> shell_mesh{
      {3_st, 4_st, 7_st},
      {Spectral::Basis::Legendre, Spectral::Basis::SphericalHarmonic,
       Spectral::Basis::SphericalHarmonic},
      {Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::Equiangular,
       Spectral::Quadrature::Equiangular}};
  const size_t n_r = shell_mesh.extents(0);
  const size_t l_max = shell_mesh.extents(1) - 1;

  const Scalar<DataVector> scalar_tensor{
      DataVector(shell_mesh.number_of_grid_points(), 1.0)};
  const auto scalar_buffer =
      PowerMonitors::shell_power_monitor_buffer(scalar_tensor, shell_mesh);
  CHECK(scalar_buffer.angular_counts ==
        scalar_tensor_angular_counts(n_r, l_max));

  tnsr::i<DataVector, 3> vector_tensor(shell_mesh.number_of_grid_points(), 1.0);
  const auto vector_buffer =
      PowerMonitors::shell_power_monitor_buffer(vector_tensor, shell_mesh);
  CHECK(vector_buffer.angular_counts ==
        vector_tensor_angular_counts(n_r, l_max));

  tnsr::ii<DataVector, 3> symmetric_rank2_tensor(
      shell_mesh.number_of_grid_points(), 1.0);
  const auto symmetric_rank2_buffer = PowerMonitors::shell_power_monitor_buffer(
      symmetric_rank2_tensor, shell_mesh);
  CHECK(symmetric_rank2_buffer.angular_counts ==
        symmetric_rank2_tensor_angular_counts(n_r, l_max));
}

template <typename TensorType>
void fill_nontrivial_shell_tensor(const gsl::not_null<TensorType*> tensor,
                                  const Mesh<3>& shell_mesh) {
  const auto logical_coords = logical_coordinates(shell_mesh);
  const DataVector& xi = logical_coords.get(0);
  const DataVector& eta = logical_coords.get(1);
  const DataVector& zeta = logical_coords.get(2);
  for (size_t component = 0; component < tensor->size(); ++component) {
    (*tensor)[component] =
        DataVector(shell_mesh.number_of_grid_points(),
                   1.0 + 0.2 * static_cast<double>(component)) +
        (1.0 + static_cast<double>(component)) * xi +
        (0.5 + 0.1 * static_cast<double>(component)) * eta +
        (0.25 - 0.05 * static_cast<double>(component)) * zeta +
        0.125 * xi * eta - 0.2 * eta * zeta +
        (0.05 + 0.01 * static_cast<double>(component)) * xi * zeta;
  }
}

template <typename TensorType>
int tensor_component_spin_weight(const size_t component) {
  if constexpr (TensorType::rank() == 0) {
    return 0;
  } else {
    const auto tensor_index =
        convert_to_cpp20_array(TensorType::get_tensor_index(component));
    const auto basis_vectors =
        ylm::TensorYlm::helpers::to_sphere_basis_vector(tensor_index);
    return std::accumulate(
        basis_vectors.begin(), basis_vectors.end(), 0,
        [](const int running_spin,
           const ylm::TensorYlm::helpers::BasisVector basis_vector) {
          return running_spin + ylm::TensorYlm::helpers::bv_to_s(basis_vector);
        });
  }
}

template <typename TensorType>
PowerMonitors::ShellPowerMonitorBuffer
expected_tensor_shell_power_monitor_buffer(const TensorType& tensor,
                                           const Mesh<3>& shell_mesh) {
  auto buffer = PowerMonitors::shell_power_monitor_buffer(
      make_with_value<DataVector>(tensor[0], 0.0), shell_mesh);
  buffer.angular_sums = 0.0;
  std::fill(buffer.angular_counts.begin(), buffer.angular_counts.end(), 0);

  if constexpr (TensorType::rank() == 0) {
    const auto scalar_buffer =
        PowerMonitors::shell_power_monitor_buffer(get(tensor), shell_mesh);
    buffer.angular_sums = scalar_buffer.angular_sums;
    buffer.angular_counts = scalar_buffer.angular_counts;
    return buffer;
  }

  const size_t l_max = shell_mesh.extents(1) - 1;
  const size_t m_max = (shell_mesh.extents(2) - 1) / 2;
  const size_t n_r = shell_mesh.extents(0);
  const ylm::Spherepack spherepack(l_max, m_max);

  auto scalar_ylm_coefficients = make_with_value<TensorType>(tensor, 0.0);
  for (size_t component = 0; component < tensor.size(); ++component) {
    scalar_ylm_coefficients[component] =
        spherepack.phys_to_spec_all_offsets(tensor[component], n_r);
  }
  const auto tensor_ylm_coefficients =
      ylm::TensorYlm::scalar_to_tensor_ylm_coefficients(
          scalar_ylm_coefficients, l_max, n_r,
          ylm::TensorYlm::CoefficientNormalization::Spherepack);

  for (size_t component = 0; component < tensor_ylm_coefficients.size();
       ++component) {
    const int spin_weight = tensor_component_spin_weight<TensorType>(component);
    const size_t abs_spin_weight = static_cast<size_t>(abs(spin_weight));
    for (ylm::SpherepackIterator it(l_max, m_max, 1, false); it; ++it) {
      const bool keep_low_l_m0_real =
          spin_weight < 0 and it.l() == 0 and it.m() == 0 and
          it.coefficient_array() ==
              ylm::SpherepackIterator::CoefficientArray::a;
      if (it.l() < abs_spin_weight and not keep_low_l_m0_real) {
        continue;
      }
      for (size_t r = 0; r < n_r; ++r) {
        buffer.angular_sums[it.l()] +=
            square(tensor_ylm_coefficients[component][it() * n_r + r]);
        ++(buffer.angular_counts[it.l()]);
      }
    }
  }
  return buffer;
}

template <typename TensorType>
void test_tensor_shell_angular_power_monitor_impl() {
  const Mesh<3> shell_mesh{
      {3_st, 4_st, 7_st},
      {Spectral::Basis::Legendre, Spectral::Basis::SphericalHarmonic,
       Spectral::Basis::SphericalHarmonic},
      {Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::Equiangular,
       Spectral::Quadrature::Equiangular}};

  TensorType tensor(shell_mesh.number_of_grid_points(), 0.0);
  fill_nontrivial_shell_tensor(make_not_null(&tensor), shell_mesh);

  const auto buffer =
      PowerMonitors::shell_power_monitor_buffer(tensor, shell_mesh);
  const auto expected_buffer =
      expected_tensor_shell_power_monitor_buffer(tensor, shell_mesh);
  const auto monitors = PowerMonitors::shell_power_monitors(tensor, shell_mesh);
  const auto expected_monitors =
      PowerMonitors::finalize_shell_power_monitor_buffer(expected_buffer);

  CHECK_ITERABLE_APPROX(buffer.angular_sums, expected_buffer.angular_sums);
  CHECK(buffer.angular_counts == expected_buffer.angular_counts);
  CHECK_ITERABLE_APPROX(monitors.angular, expected_monitors.angular);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.LinearOperators.PowerMonitors",
                  "[NumericalAlgorithms][LinearOperators][Unit]") {
  test_power_monitors_impl();
  test_power_monitors_second_impl();
  test_relative_truncation_error_impl();
  test_relative_truncation_error_with_symmetry();
  test_relative_truncation_error_linear_function();
  test_convergence_rate();
  test_pile_up_modes();
  test_shell_power_monitors();
  test_shell_tensor_count_semantics();
  test_tensor_shell_angular_power_monitor_impl<Scalar<DataVector>>();
  test_tensor_shell_angular_power_monitor_impl<tnsr::i<DataVector, 3>>();
  test_tensor_shell_angular_power_monitor_impl<tnsr::ii<DataVector, 3>>();
  test_tensor_shell_angular_power_monitor_impl<tnsr::ijj<DataVector, 3>>();
}
