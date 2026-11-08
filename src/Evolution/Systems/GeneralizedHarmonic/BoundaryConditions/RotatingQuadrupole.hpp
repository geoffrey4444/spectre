// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>
#include <pup.h>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/Auto.hpp"
#include "Options/Context.hpp"
#include "Options/String.hpp"
#include "Utilities/TMPL.hpp"

namespace gh::BoundaryConditions {
/*!
 * \brief A smoothly switched monochromatic, rotating quadrupolar incoming
 * curvature tensor for ConstraintPreservingBjorhus.
 *
 * The nonzero Cartesian components are
 * \f$Q_{xx}=-Q_{yy}=A W(t)\cos\vartheta\f$ and
 * \f$Q_{xy}=Q_{yx}=A W(t)\sin\vartheta\f$, with
 * \f$\vartheta=\omega(t-t_0)+\phi_0\f$. The boundary condition projects
 * this tensor into its physical, transverse trace-free subspace and prescribes
 * it as the incoming Weyl characteristic \f$U^{3-}_{ab}\f$.
 * Thus `Amplitude` is a curvature amplitude (inverse length squared), not a
 * strain amplitude. There is no automatic radius scaling.
 *
 * On a flat spherical boundary this is an electric-parity quadrupole with
 * angular dependence \f$2\varphi-\omega t\f$. Positive `AngularFrequency`
 * carries positive z angular momentum (pattern speed \f$\omega/2\f$).
 * For a weak monochromatic wave, \f$dJ_z/dE=2/\omega\f$. Absorption by a
 * Kerr hole spinning about +z and increasing its dimensionless spin require,
 * for positive frequency, respectively,
 * \f$\omega>2\Omega_H\f$ and \f$\omega<1/(\chi M)\f$. These conditions
 * change as the hole evolves and do not imply that a fixed-frequency drive
 * overspins it. Finite ramps introduce sidebands; curvature and the finite
 * boundary radius also prevent interpreting this as an exact Kerr scattering
 * solution.
 *
 * The envelope uses the quintic smoothstep \f$S(x)=x^3(10-15x+6x^2)\f$:
 * zero before `StartTime`, S during `RampUpDuration`, and one thereafter.
 * If `StopTime` is set, it begins a ramp of length `RampDownDuration` with
 * envelope 1-S, then becomes exactly zero. The ramps cannot overlap. The
 * resulting envelope and its first two derivatives are continuous.
 */
class RotatingQuadrupole {
 public:
  struct Amplitude {
    using type = double;
    static constexpr Options::String help{
        "Nonnegative incoming Weyl characteristic amplitude (1/length^2)."};
    static double lower_bound() { return 0.; }
  };
  struct AngularFrequency {
    using type = double;
    static constexpr Options::String help{
        "Signed nonzero angular frequency in inertial time; positive rotates "
        "about +z with m=2."};
  };
  struct Phase {
    using type = double;
    static constexpr Options::String help{
        "Carrier phase at StartTime (radians)."};
  };
  struct StartTime {
    using type = double;
    static constexpr Options::String help{
        "Inertial time when the ramp-up begins."};
  };
  struct RampUpDuration {
    using type = double;
    static constexpr Options::String help{
        "Strictly positive ramp-up duration."};
  };
  struct StopTime {
    using type = Options::Auto<double, Options::AutoLabel::None>;
    static constexpr Options::String help{
        "Time when ramp-down begins, or None to drive indefinitely."};
  };
  struct RampDownDuration {
    using type = double;
    static constexpr Options::String help{
        "Strictly positive ramp-down duration (unused if StopTime is None)."};
  };
  using options = tmpl::list<Amplitude, AngularFrequency, Phase, StartTime,
                             RampUpDuration, StopTime, RampDownDuration>;
  static constexpr Options::String help{
      "A rotating m=2 incoming curvature wave with smooth finite ramps."};

  RotatingQuadrupole() = default;
  RotatingQuadrupole(double amplitude, double angular_frequency, double phase,
                     double start_time, double ramp_up_duration,
                     std::optional<double> stop_time, double ramp_down_duration,
                     const Options::Context& context = {});

  /// Return the dimensionless envelope, between zero and one.
  double envelope(double time) const;
  /// Return the unprojected Cartesian spacetime curvature tensor.
  tnsr::aa<double, 3, Frame::Inertial> operator()(double time) const;
  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

 private:
  double amplitude_ = 0.;
  double angular_frequency_ = 1.;
  double phase_ = 0.;
  double start_time_ = 0.;
  double ramp_up_duration_ = 1.;
  std::optional<double> stop_time_{};
  double ramp_down_duration_ = 1.;
};
}  // namespace gh::BoundaryConditions
