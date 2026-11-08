\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond
# Driving a Kerr black hole with a rotating wave {#example_kerr_spin_up}

The `RotatingQuadrupole` option of
gh::BoundaryConditions::ConstraintPreservingBjorhus injects an approximately
monochromatic gravitational wave with angular momentum about the inertial
z axis. It controls the physical incoming Weyl characteristic while retaining
the existing constraint and gauge boundary conditions. It does not require an
initial-data constraint solve for a wave packet already inside the domain.

## Boundary prescription

The Cartesian source tensor has nonzero components

\f{align}
 Q_{xx}=-Q_{yy}&=A W(t)\cos\vartheta, &
 Q_{xy}=Q_{yx}&=A W(t)\sin\vartheta, \\
 \vartheta&=\omega(t-t_0)+\phi_0.
\f}

The physical boundary projector is
\f$\mathcal P_{ab}{}^{cd}=P^c_aP^d_b-P_{ab}P^{cd}/2\f$.
The implementation replaces the homogeneous incoming-curvature target by
\f$U^{3-}_{ab}=\mathcal P_{ab}{}^{cd}Q_{cd}\f$. Specifically, it adds
\f$-\lambda_-\mathcal P_{ab}{}^{cd}Q_{cd}\f$ to the incoming GH
characteristic time-derivative correction. This is the inhomogeneous version
of the Weyl boundary condition in
[Lindblom et al., Eqs. (66)-(68)](https://arxiv.org/pdf/gr-qc/0512093).

`Amplitude` is the coefficient \f$A\f$ of the incoming Weyl characteristic,
with units of inverse length squared. It is neither strain nor injected
energy. It is specified at the boundary without automatic radius scaling.
`AngularFrequency` is \f$\omega\f$, in radians per inertial coordinate time.
The tensor pattern rotates at half this frequency.

For a flat spherical boundary, projecting onto its orthonormal angular basis
gives

\f{align}
 Q_+&=\frac{A W}{2}(1+\cos^2\theta)
       \cos(2\varphi-\vartheta), \\
 Q_\times&=-A W\cos\theta\sin(2\varphi-\vartheta).
\f}

Thus positive \f$\omega\f$ gives a prograde electric quadrupole for a hole
spinning about +z. The positive-frequency part has \f$m=2\f$; reality adds
the \f$(-m,-\omega)\f$ partner, which has the same angular momentum per unit
energy. Reversing the frequency reverses the rotation. The old axisymmetric
`IncomingWaveProfile` source does not provide this rotating pattern.

The envelope uses the quintic smoothstep
\f$S(x)=x^3(10-15x+6x^2)\f$: it is zero before `StartTime`, rises to one over
`RampUpDuration`, and remains one until `StopTime`. If `StopTime` is `None`,
the drive continues indefinitely. Otherwise it falls to zero over
`RampDownDuration`. The envelope and its first two derivatives are continuous.
Finite ramps necessarily introduce frequencies away from the carrier.

For example, in units where the initial hole mass is one:

```yaml
ConstraintPreservingBjorhus:
  Type: ConstraintPreservingPhysical
  RotatingQuadrupole:
    Amplitude: 0.00026
    AngularFrequency: 0.9090909090909091
    Phase: 0.0
    StartTime: 0.0
    RampUpDuration: 150.0
    StopTime: None
    RampDownDuration: 150.0
```

`RotatingQuadrupole` and `IncomingWaveProfile` are alternative options;
specify only one of them.

## Choosing the frequency

For weak monochromatic scattering,
\f$dJ_z/dE=m/\omega\f$. Absorption changes sign at the superradiant threshold
\f$\omega=m\Omega_H\f$, where
\f$\Omega_H=\chi/[2M(1+\sqrt{1-\chi^2})]\f$.
These relations and their use in nonlinear gravitational-wave scattering
are discussed by
[East, Ramazanoglu, and Pretorius](https://arxiv.org/pdf/1312.4529).
For positive absorbed mass and a prograde \f$m=2\f$ wave,

\f{align}
 \frac{d\chi}{dt}
   &=\frac{\dot M}{M^2}\left(\frac{2}{\omega}-2\chi M\right), \\
 2\Omega_H&<\omega<\frac{1}{\chi M}
\f}

is the instantaneous absorption-and-spin-up window. At \f$M=1\f$ and
\f$\chi=0.99\f$, it is approximately
\f$0.8676087<\omega<1.0101010\f$. Both bounds change as the hole evolves.

This window differs from the narrower test-body "overspin" window. Assuming
\f$\Delta J=2\Delta M/\omega\f$ throughout absorption and ignoring the
changing absorption probability gives

\f$
 J(M)=\chi_0M_0^2+\frac{2}{\omega}(M-M_0).
\f$

The line tangent to \f$J=M^2\f$ has
\f$M_0\omega=1/(1+\sqrt{1-\chi_0})\f$. For \f$\chi_0=0.99\f$, this is
\f$M_0\omega=0.9090909\f$, with tangency at \f$M/M_0=1.1\f$.
Lower frequencies encounter the superradiant threshold earlier; higher
frequencies reach a subextremal maximum of \f$\chi\f$ and then spin down
under continued absorption. Derived weak, adiabatic estimates are:

| \f$M_0\omega\f$ | \f$M/M_0\f$ at threshold | \f$\chi\f$ at threshold |
| --- | --- | --- |
| 0.888 | 1.0223157 | 0.9953414 |
| 0.900 | 1.0446350 | 0.9981000 |
| 0.9090909 | 1.1000000 | 1.0000000 (limiting tangency) |

These are model predictions, not simulation results or a prescription that
guarantees extremality. A weak drive loses absorption efficiency near the
threshold. Second-order effects also invalidate a test-body claim of
overspinning; see
[Sorce and Wald](https://arxiv.org/abs/1707.05862).

## Amplitude and verification

In the flat, locally planar limit an ingoing TT strain obeys
\f$U^{3-}_{ij}=-2\partial_t^2h^{\mathrm{TT}}_{ij}\f$.
During the plateau its strain coefficient is therefore
\f$B=A/(2\omega^2)\f$. Integrating the quadrupole angular pattern gives the
leading far-zone, time-averaged incident luminosity

\f$
 \left\langle\dot E_{\mathrm{in}}\right\rangle
   \simeq\frac{R^2 A^2}{40\omega^2},\qquad
 \left\langle\dot J_{z,\mathrm{in}}\right\rangle
   \simeq\frac{2}{\omega}
     \left\langle\dot E_{\mathrm{in}}\right\rangle.
\f$

For \f$R=200M_0\f$, \f$A=2.6\times10^{-4}/M_0^2\f$, and
\f$M_0\omega=0.90909\f$, the incident power is about
\f$8.2\times10^{-5}\f$. Accreting \f$0.1M_0\f$ would take at least
\f$1220M_0\f$ at this plateau power, and longer with incomplete absorption
or ramping. At fixed radius halving the amplitude increases this estimate
by a factor of four.

Use analytic Kerr-Schild gauge source functions from the initial hole to
retain that gauge prescription. The evolving geometry is no longer exactly
Kerr-Schild. Shape and size control must track the distorted horizon and keep
the excision boundary an outflow boundary.

The `DrivenKerrSchild.yaml` example uses 29 radial shells out to
\f$200M_0\f$, with angular resolution decreasing from \f$L=32\f$ near the
hole to \f$L=16\f$ in the wave zone. `NumGridPointsAndGridSpacing` weights
the element distribution for local time stepping. Starting the smooth ramp at
zero still leaves approximately a light-crossing time before the injected
wave reaches the hole; boundary turn-on time is not horizon arrival time.

For stationary startup, initialize the excision surface as a homothetic copy
of the horizon, matching the shape controller's target. With grid inner radius
\f$r_{\mathrm{in}}\f$, set only `ShapeMap.InitialValues.Mass` to
\f$r_{\mathrm{in}}/(1+\sqrt{1-\chi_0^2})\f$ and use `SizeInitialValues: Auto`.
Keep the physical initial-data and gauge masses at \f$M_0\f$. For
\f$r_{\mathrm{in}}=1.06M_0\f$ and \f$\chi_0=0.99\f$, this map parameter is
\f$0.9289547991M_0\f$. Using the physical mass for this map initializer instead
produces a constant Boyer-Lindquist-radius ellipsoid, which the shape
controller immediately deforms even in stationary Kerr. Resolution and
positive excision characteristic speeds still need to be checked numerically.

A finite spherical boundary and this spherical quadrupole are not an exact
Kerr spheroidal scattering solution. Curvature, reflection, transients, and
nonlinear mode coupling affect the measured response. Compare an undriven
run at identical resolution, and check convergence of horizon mass, angular
momentum, dimensionless spin, constraints, and excision characteristic speeds.
Early apparent-horizon spin oscillations alone do not establish a lasting
spin increase. Ramp down and allow the remnant to settle before interpreting
a final spin.

The reported `DimensionlessSpinMagnitude` cannot by itself diagnose
superextremality. The horizon observations compute
\f$M_{\mathrm{irr}}^2=A_H/(16\pi)\f$ and the Christodoulou mass
\f$M_C^2=M_{\mathrm{irr}}^2+J^2/(4M_{\mathrm{irr}}^2)\f$ from the same
approximate-Killing-vector spin magnitude \f$J\f$. Consequently,

\f$
 \zeta=\frac{J}{2M_{\mathrm{irr}}^2}=\frac{8\pi J}{A_H},\qquad
 \chi_{\mathrm{reported}}=\frac{J}{M_C^2}
   =\frac{2\zeta}{1+\zeta^2}\leq1.
\f$

The last bound is an algebraic property of the diagnostic, even if
\f$\zeta>1\f$; the map gives the same reported spin for \f$\zeta\f$ and
\f$1/\zeta\f$. Track \f$\zeta\f$ directly using the observed area and
dimensionful spin-vector magnitude, together with \f$J_z\f$ to verify the
rotation direction. Follow horizon persistence and search for enclosing
marginally trapped surfaces if an inner surface approaches or exceeds
\f$\zeta=1\f$. In a strongly distorted, nonaxisymmetric geometry,
\f$\zeta\f$ remains an approximate-Killing-vector diagnostic; alternative
extremality measures and enclosing horizons are discussed by
[Lovelace et al.](https://arxiv.org/pdf/1411.7297).
