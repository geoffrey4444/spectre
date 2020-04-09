// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <random>
#include <utility>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"        // IWYU pragma: keep
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Characteristics.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Equations.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/DataStructures/RandomUnitNormal.hpp"
#include "Helpers/NumericalAlgorithms/DiscontinuousGalerkin/NumericalFluxes/TestHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NumericalFluxes/NumericalFluxHelpers.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeSpacetimeQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

// IWYU pragma: no_forward_declare GeneralizedHarmonic::Tags::Pi
// IWYU pragma: no_forward_declare GeneralizedHarmonic::Tags::Phi
// IWYU pragma: no_forward_declare GeneralizedHarmonic::Tags::UPsi
// IWYU pragma: no_forward_declare GeneralizedHarmonic::Tags::UZero
// IWYU pragma: no_forward_declare GeneralizedHarmonic::Tags::UMinus
// IWYU pragma: no_forward_declare GeneralizedHarmonic::Tags::UPlus
// IWYU pragma: no_forward_declare Tags::CharSpeed
// IWYU pragma: no_forward_declare Tags::dt
// IWYU pragma: no_forward_declare Tensor
// IWYU pragma: no_forward_declare Variables

namespace GeneralizedHarmonic {
namespace GeneralizedHarmonic_detail {
template <typename FieldTag>
db::const_item_type<FieldTag> weight_char_field_upwind_multipenalty(
    const db::const_item_type<FieldTag>& char_field,
    const Scalar<DataVector>& char_speed) noexcept;
}  // namespace GeneralizedHarmonic_detail
}  // namespace GeneralizedHarmonic

namespace {
// Test GH upwind flux using random fields
void test_upwind_flux_multipenalty_random() noexcept {
  constexpr size_t spatial_dim = 3;
  const DataVector used_for_size{5,
                                 std::numeric_limits<double>::signaling_NaN()};

  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> dist(-1.0, 1.0);
  const auto nn_generator = make_not_null(&generator);
  const auto nn_dist = make_not_null(&dist);

  const auto one = make_with_value<Scalar<DataVector>>(used_for_size, 1.0);
  const auto minus_five =
      make_with_value<Scalar<DataVector>>(used_for_size, -5.0);

  // Choose random characteristic fields, characteristic speeds, gamma2, and
  // normal vector (note: no need to ensure normal vector is actually a unit
  // normal vector for this check of the upwind multipenalty flux's consistency)
  const auto u_psi = make_with_random_values<
      tnsr::aa<DataVector, spatial_dim, Frame::Inertial>>(nn_generator, nn_dist,
                                                          used_for_size);
  const auto u_zero = make_with_random_values<
      tnsr::iaa<DataVector, spatial_dim, Frame::Inertial>>(
      nn_generator, nn_dist, used_for_size);
  const auto u_plus = make_with_random_values<
      tnsr::aa<DataVector, spatial_dim, Frame::Inertial>>(nn_generator, nn_dist,
                                                          used_for_size);
  const auto u_minus = make_with_random_values<
      tnsr::aa<DataVector, spatial_dim, Frame::Inertial>>(nn_generator, nn_dist,
                                                          used_for_size);

  const auto char_speed_u_psi = make_with_random_values<Scalar<DataVector>>(
      nn_generator, nn_dist, used_for_size);
  const auto char_speed_u_zero = make_with_random_values<Scalar<DataVector>>(
      nn_generator, nn_dist, used_for_size);
  const auto char_speed_u_plus = make_with_random_values<Scalar<DataVector>>(
      nn_generator, nn_dist, used_for_size);
  const auto char_speed_u_minus = make_with_random_values<Scalar<DataVector>>(
      nn_generator, nn_dist, used_for_size);
  const std::array<DataVector, 4> char_speeds{
      {get(char_speed_u_psi), get(char_speed_u_zero), get(char_speed_u_plus),
       get(char_speed_u_minus)}};

  // Compute what the speeds would have been with the opposite normal
  const std::array<DataVector, 4> char_speeds_minus_normal{
      {-get(char_speed_u_psi), -get(char_speed_u_zero),
       -get(char_speed_u_minus), -get(char_speed_u_plus)}};

  const auto gamma_2 = make_with_random_values<Scalar<DataVector>>(
      nn_generator, nn_dist, used_for_size);
  const auto normal = make_with_random_values<
      tnsr::i<DataVector, spatial_dim, Frame::Inertial>>(nn_generator, nn_dist,
                                                         used_for_size);
  tnsr::i<DataVector, spatial_dim, Frame::Inertial> minus_normal = normal;
  for (size_t i = 0; i < spatial_dim; ++i) {
    minus_normal.get(i) *= -1.0;
  }

  GeneralizedHarmonic::UpwindMultipenaltyFlux<spatial_dim> flux_computer{};

  INFO("test generalized-harmonic upwind weighting function")
  // If all the char speeds are +1, the weighted fields should vanish
  const auto weighted_u_zero_one = GeneralizedHarmonic::
      GeneralizedHarmonic_detail::weight_char_field_upwind_multipenalty<
          GeneralizedHarmonic::Tags::UZero<spatial_dim, Frame::Inertial>>(
          u_zero, one);
  const auto zero_iaa =
      make_with_value<tnsr::iaa<DataVector, spatial_dim, Frame::Inertial>>(
          used_for_size, 0.0);
  CHECK_ITERABLE_APPROX(weighted_u_zero_one, zero_iaa);

  // If all the char speeds are -5, the weighted fields should just be
  // -5 * the original fields
  const auto weighted_u_plus_minus_five = GeneralizedHarmonic::
      GeneralizedHarmonic_detail::weight_char_field_upwind_multipenalty<
          GeneralizedHarmonic::Tags::UPlus<spatial_dim, Frame::Inertial>>(
          u_plus, minus_five);
  auto u_plus_times_minus_five = u_plus;
  for (auto it = u_plus_times_minus_five.begin();
       it != u_plus_times_minus_five.end(); ++it) {
    *it *= -5.0;
  }
  CHECK_ITERABLE_APPROX(weighted_u_plus_minus_five, u_plus_times_minus_five);

  INFO("test consistency of the generalized-harmonic upwind multipenalty flux")
  // Check that if the same quantities are given for the interior and exterior
  // that the numerical multipenalty flux vanishes
  auto packaged_data = ::TestHelpers::NumericalFluxes::get_packaged_data(
      flux_computer, used_for_size, u_psi, u_zero, u_plus, u_minus, char_speeds,
      gamma_2, normal);
  auto packaged_data_minus_normal =
      ::TestHelpers::NumericalFluxes::get_packaged_data(
          flux_computer, used_for_size, u_psi, u_zero, u_minus, u_plus,
          char_speeds_minus_normal, gamma_2, minus_normal);

  auto psi_normal_dot_numerical_flux =
      make_with_value<tnsr::aa<DataVector, spatial_dim, Frame::Inertial>>(
          used_for_size, std::numeric_limits<double>::signaling_NaN());
  auto pi_normal_dot_numerical_flux =
      make_with_value<tnsr::aa<DataVector, spatial_dim, Frame::Inertial>>(
          used_for_size, std::numeric_limits<double>::signaling_NaN());
  auto phi_normal_dot_numerical_flux =
      make_with_value<tnsr::iaa<DataVector, spatial_dim, Frame::Inertial>>(
          used_for_size, std::numeric_limits<double>::signaling_NaN());
  dg::NumericalFluxes::normal_dot_numerical_fluxes(
      flux_computer, packaged_data, packaged_data_minus_normal,
      make_not_null(&psi_normal_dot_numerical_flux),
      make_not_null(&pi_normal_dot_numerical_flux),
      make_not_null(&phi_normal_dot_numerical_flux));

  auto psi_normal_dot_numerical_flux_expected =
      make_with_value<tnsr::aa<DataVector, spatial_dim, Frame::Inertial>>(
          used_for_size, 0.0);
  auto pi_normal_dot_numerical_flux_expected =
      make_with_value<tnsr::aa<DataVector, spatial_dim, Frame::Inertial>>(
          used_for_size, 0.0);
  auto phi_normal_dot_numerical_flux_expected =
      make_with_value<tnsr::iaa<DataVector, spatial_dim, Frame::Inertial>>(
          used_for_size, 0.0);

  CHECK_ITERABLE_APPROX(psi_normal_dot_numerical_flux,
                        psi_normal_dot_numerical_flux_expected);
  CHECK_ITERABLE_APPROX(pi_normal_dot_numerical_flux,
                        pi_normal_dot_numerical_flux_expected);
  CHECK_ITERABLE_APPROX(phi_normal_dot_numerical_flux,
                        phi_normal_dot_numerical_flux_expected);
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.GeneralizedHarmonic.UpwindMultipenaltyFlux",
    "[Unit][Evolution]") {
  test_upwind_flux_multipenalty_random();
}
