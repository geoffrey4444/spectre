// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>

#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/AnalyticChristoffel.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/DampedHarmonic.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/DhGaugeParameters.hpp"
#include "Options/Context.hpp"

namespace gh::gauges {
/*!
 * \brief Roll from a stationary analytic gauge to damped harmonic gauge.
 *
 * Uses the existing `damped_harmonic_rollon` temporal weight. The initial
 * analytic gauge is reevaluated at the current inertial coordinates, and its
 * derivatives are inertial derivatives. In particular, initial values are not
 * frozen on moving grid points. This permits moving meshes while retaining
 * AnalyticChristoffel's restriction to stationary analytic gauge sources.
 *
 * This changes the gauge history; it does not change damped harmonic gauge
 * after the initial contribution has decayed.
 */
class DampedHarmonicRollOn final : public GaugeCondition {
 public:
  struct InitialGauge {
    using type = AnalyticChristoffel;
    static constexpr Options::String help{
        "Stationary analytic gauge from which to transition."};
  };

  using options =
      tmpl::push_front<DhGaugeParameters<true>::options, InitialGauge>;
  static constexpr Options::String help{
      "Smoothly transition from a stationary analytic gauge to damped "
      "harmonic gauge using R=1-exp(-((t-t0)/window)^4)."};

  DampedHarmonicRollOn() = default;
  DampedHarmonicRollOn(AnalyticChristoffel initial_gauge, double start,
                       double window, double width,
                       const std::array<double, 3>& amplitudes,
                       const std::array<int, 3>& exponents,
                       const Options::Context& context = {});

  /// \cond
  explicit DampedHarmonicRollOn(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(DampedHarmonicRollOn);  // NOLINT
  /// \endcond

  std::unique_ptr<GaugeCondition> get_clone() const override;
  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

  template <size_t Dim, class AllSolutionsForChristoffelAnalytic>
  void gauge_and_spacetime_derivative(
      const gsl::not_null<tnsr::a<DataVector, Dim, Frame::Inertial>*> gauge_h,
      const gsl::not_null<tnsr::ab<DataVector, Dim, Frame::Inertial>*>
          d4_gauge_h,
      const Scalar<DataVector>& lapse,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& shift,
      const Scalar<DataVector>& sqrt_det_spatial_metric,
      const tnsr::II<DataVector, Dim, Frame::Inertial>& inverse_spatial_metric,
      const tnsr::abb<DataVector, Dim, Frame::Inertial>& d4_spacetime_metric,
      const Scalar<DataVector>& half_pi_two_normals,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& half_phi_two_normals,
      const tnsr::aa<DataVector, Dim, Frame::Inertial>& spacetime_metric,
      const tnsr::iaa<DataVector, Dim, Frame::Inertial>& phi,
      const Mesh<Dim>& mesh, const double time,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& inertial_coords,
      const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                            Frame::Inertial>& inverse_jacobian,
      const AllSolutionsForChristoffelAnalytic meta) const {
    if (time <= parameters_.rollon_start) {
      initial_gauge_.gauge_and_spacetime_derivative(gauge_h, d4_gauge_h, mesh,
                                                    time, inertial_coords,
                                                    inverse_jacobian, meta);
      return;
    }
    tnsr::a<DataVector, Dim, Frame::Inertial> initial_h{get(lapse).size()};
    tnsr::ab<DataVector, Dim, Frame::Inertial> initial_dh{get(lapse).size()};
    initial_gauge_.gauge_and_spacetime_derivative(
        make_not_null(&initial_h), make_not_null(&initial_dh), mesh, time,
        inertial_coords, inverse_jacobian, meta);
    damped_harmonic_rollon(
        gauge_h, d4_gauge_h, initial_h, initial_dh, lapse, shift,
        sqrt_det_spatial_metric, inverse_spatial_metric, d4_spacetime_metric,
        half_pi_two_normals, half_phi_two_normals, spacetime_metric, phi, time,
        inertial_coords, parameters_.amplitudes[0], parameters_.amplitudes[1],
        parameters_.amplitudes[2], parameters_.exponents[0],
        parameters_.exponents[1], parameters_.exponents[2],
        parameters_.rollon_start, parameters_.rollon_window,
        parameters_.spatial_decay_width);
  }

 private:
  AnalyticChristoffel initial_gauge_{};
  DhGaugeParameters<true> parameters_{};
};
}  // namespace gh::gauges
