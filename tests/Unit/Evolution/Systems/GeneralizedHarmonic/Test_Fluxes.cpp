// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <random>
#include <utility>

#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Characteristics.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Equations.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/DataStructures/RandomUnitNormal.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeSpacetimeQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

// IWYU pragma: no_forward_declare Tensor
// IWYU pragma: no_forward_declare Variables
namespace GeneralizedHarmonic {
namespace GeneralizedHarmonic_detail {
template <typename FieldTag>
db::const_item_type<FieldTag> weight_char_field_by_char_speed(
    const db::const_item_type<FieldTag>& char_field,
    const Scalar<DataVector>& char_speed) noexcept;
}  // namespace GeneralizedHarmonic_detail
}  // namespace GeneralizedHarmonic

namespace {
template <size_t Dim, typename DataType>
void test_gh_fluxes(const DataType& used_for_size) {
  pypp::check_with_random_values<1>(
      &GeneralizedHarmonic::ComputeNormalDotFluxes<Dim>::apply, "TestFunctions",
      {"spacetime_metric_normal_dot_flux", "pi_normal_dot_flux",
       "phi_dot_flux"},
      {{{-1.0, 1.0}}}, used_for_size);
}

template <size_t Dim>
void test_gh_fluxes_from_char_speeds_random() noexcept {
  constexpr size_t spatial_dim = Dim;
  const DataVector used_for_size{5,
                                 std::numeric_limits<double>::signaling_NaN()};

  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> dist(-1.0, 1.0);
  std::uniform_real_distribution<> dist_pert(-0.1, 0.1);
  const auto nn_generator = make_not_null(&generator);
  const auto nn_dist = make_not_null(&dist);
  const auto nn_dist_pert = make_not_null(&dist_pert);

  const auto one = make_with_value<Scalar<DataVector>>(used_for_size, 1.0);
  const auto minus_one =
      make_with_value<Scalar<DataVector>>(used_for_size, -1.0);
  const auto minus_five =
      make_with_value<Scalar<DataVector>>(used_for_size, -5.0);

  // Choose spacetime_metric randomly, but make sure the result is
  // still invertible. To do this, start with
  // Minkowski, and then add a 10% random perturbation.
  auto spacetime_metric = make_with_random_values<
      tnsr::aa<DataVector, spatial_dim, Frame::Inertial>>(
      nn_generator, nn_dist_pert, used_for_size);
  get<0, 0>(spacetime_metric) += get(minus_one);
  for (size_t i = 1; i < spatial_dim + 1; ++i) {
    spacetime_metric.get(i, i) += get(one);
  }

  // Set pi, phi to be random (phi, pi should not need to be consistent with
  // spacetime_metric for the flux consistency tests to pass)
  const auto phi = make_with_random_values<
      tnsr::iaa<DataVector, spatial_dim, Frame::Inertial>>(
      nn_generator, nn_dist_pert, used_for_size);
  const auto pi = make_with_random_values<
      tnsr::aa<DataVector, spatial_dim, Frame::Inertial>>(
      nn_generator, nn_dist_pert, used_for_size);

  const auto spatial_metric = gr::spatial_metric(spacetime_metric);
  const auto inverse_spatial_metric =
      determinant_and_inverse(spatial_metric).second;
  const tnsr::i<DataVector, spatial_dim, Frame::Inertial> unit_normal_one_form =
      raise_or_lower_index(random_unit_normal(nn_generator, spatial_metric),
                           spatial_metric);

  const auto gamma_1 = make_with_random_values<Scalar<DataVector>>(
      nn_generator, nn_dist, used_for_size);
  const auto gamma_2 = make_with_random_values<Scalar<DataVector>>(
      nn_generator, nn_dist, used_for_size);

  const auto shift = gr::shift(spacetime_metric, inverse_spatial_metric);
  const auto lapse = gr::lapse(shift, spacetime_metric);

  // Get the characteristic fields and speeds
  const auto char_fields = GeneralizedHarmonic::characteristic_fields(
      gamma_2, inverse_spatial_metric, spacetime_metric, pi, phi,
      unit_normal_one_form);
  const auto& u_psi =
      get<GeneralizedHarmonic::Tags::UPsi<Dim, Frame::Inertial>>(char_fields);
  const auto& u_zero =
      get<GeneralizedHarmonic::Tags::UZero<Dim, Frame::Inertial>>(char_fields);
  const auto& u_plus =
      get<GeneralizedHarmonic::Tags::UPlus<Dim, Frame::Inertial>>(char_fields);
  const auto& u_minus =
      get<GeneralizedHarmonic::Tags::UMinus<Dim, Frame::Inertial>>(char_fields);
  const auto char_speeds = GeneralizedHarmonic::characteristic_speeds(
      gamma_1, lapse, shift, unit_normal_one_form);

  std::array<DataVector, 4> char_speeds_one{
      {get(one), get(one), get(one), get(one)}};
  std::array<DataVector, 4> char_speeds_minus_five{
      {get(minus_five), get(minus_five), get(minus_five), get(minus_five)}};

  // If all the char speeds are +1, the weighted fields should vanish
  const auto weighted_u_zero_one = GeneralizedHarmonic::
      GeneralizedHarmonic_detail::weight_char_field_by_char_speed<
          GeneralizedHarmonic::Tags::UZero<spatial_dim, Frame::Inertial>>(
          get<GeneralizedHarmonic::Tags::UZero<Dim, Frame::Inertial>>(
              char_fields),
          one);
  CHECK_ITERABLE_APPROX(weighted_u_zero_one, u_zero);

  // If all the char speeds are -5, the weighted fields should just be
  // -5 * the original fields
  const auto weighted_u_plus_minus_five = GeneralizedHarmonic::
      GeneralizedHarmonic_detail::weight_char_field_by_char_speed<
          GeneralizedHarmonic::Tags::UPlus<spatial_dim, Frame::Inertial>>(
          get<GeneralizedHarmonic::Tags::UPlus<Dim, Frame::Inertial>>(
              char_fields),
          minus_five);
  auto u_plus_times_minus_five = u_plus;
  for (auto it = u_plus_times_minus_five.begin();
       it != u_plus_times_minus_five.end(); ++it) {
    *it *= -5.0;
  }
  CHECK_ITERABLE_APPROX(weighted_u_plus_minus_five, u_plus_times_minus_five);

  INFO("test GH flux for char speeds agrees with GH flux from fields")

  // Check that if the same fields are given for the interior and exterior
  // (except that the normal vector gets multiplied by -1.0) that the
  // numerical flux reduces to the flux

  ::GeneralizedHarmonic::ComputeNormalDotFluxes<spatial_dim>
      normal_dot_flux_computer{};
  auto spacetime_metric_normal_dot_flux = make_with_value<
      db::item_type<::Tags::NormalDotFlux<gr::Tags::SpacetimeMetric<
          spatial_dim, Frame::Inertial, DataVector>>>>(
      used_for_size, std::numeric_limits<double>::signaling_NaN());
  auto pi_normal_dot_flux = make_with_value<db::item_type<::Tags::NormalDotFlux<
      GeneralizedHarmonic::Tags::Pi<spatial_dim, Frame::Inertial>>>>(
      used_for_size, std::numeric_limits<double>::signaling_NaN());
  auto phi_normal_dot_flux =
      make_with_value<db::item_type<::Tags::NormalDotFlux<
          GeneralizedHarmonic::Tags::Phi<spatial_dim, Frame::Inertial>>>>(
          used_for_size, std::numeric_limits<double>::signaling_NaN());
  normal_dot_flux_computer.apply(
      make_not_null(&spacetime_metric_normal_dot_flux),
      make_not_null(&pi_normal_dot_flux), make_not_null(&phi_normal_dot_flux),
      spacetime_metric, pi, phi, gamma_1, gamma_2, lapse, shift,
      inverse_spatial_metric, unit_normal_one_form);

  ::GeneralizedHarmonic::ComputeNormalDotFluxesFromCharFields<spatial_dim>
      normal_dot_flux_from_char_fields_computer{};
  auto spacetime_metric_normal_dot_flux_from_char_fields = make_with_value<
      db::item_type<::Tags::NormalDotFlux<gr::Tags::SpacetimeMetric<
          spatial_dim, Frame::Inertial, DataVector>>>>(
      used_for_size, std::numeric_limits<double>::signaling_NaN());
  auto pi_normal_dot_flux_from_char_fields =
      make_with_value<db::item_type<::Tags::NormalDotFlux<
          GeneralizedHarmonic::Tags::Pi<spatial_dim, Frame::Inertial>>>>(
          used_for_size, std::numeric_limits<double>::signaling_NaN());
  auto phi_normal_dot_flux_from_char_fields =
      make_with_value<db::item_type<::Tags::NormalDotFlux<
          GeneralizedHarmonic::Tags::Phi<spatial_dim, Frame::Inertial>>>>(
          used_for_size, std::numeric_limits<double>::signaling_NaN());
  normal_dot_flux_from_char_fields_computer.apply(
      make_not_null(&spacetime_metric_normal_dot_flux_from_char_fields),
      make_not_null(&pi_normal_dot_flux_from_char_fields),
      make_not_null(&phi_normal_dot_flux_from_char_fields), u_psi, u_zero,
      u_plus, u_minus, char_speeds, gamma_2, unit_normal_one_form);

  CHECK_ITERABLE_APPROX(spacetime_metric_normal_dot_flux_from_char_fields,
                        spacetime_metric_normal_dot_flux);
  CHECK_ITERABLE_APPROX(pi_normal_dot_flux, pi_normal_dot_flux);
  CHECK_ITERABLE_APPROX(phi_normal_dot_flux, phi_normal_dot_flux);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.GeneralizedHarmonic.NormalDotFluxes",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/GeneralizedHarmonic/"};

  GENERATE_UNINITIALIZED_DATAVECTOR;
  CHECK_FOR_DATAVECTORS(test_gh_fluxes, (1, 2, 3));
}

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.GeneralizedHarmonic.NormalDotFluxesFromCharFields",
    "[Unit][Evolution]") {
  test_gh_fluxes_from_char_speeds_random<1>();
  test_gh_fluxes_from_char_speeds_random<2>();
  test_gh_fluxes_from_char_speeds_random<3>();
}
