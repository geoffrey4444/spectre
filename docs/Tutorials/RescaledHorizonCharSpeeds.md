\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond

# Characteristic speeds between excision and the horizon {#tutorial_rescaled_horizon_char_speeds}

The apparent-horizon finder can observe characteristic speeds on a family of
scaled copies of a converged horizon. Plotting their angular minima over time
can help identify an excision size with better outflow characteristics. This
is an optional observation; it does not change the excision surface or the
control system.

## Enable the observation

Add `RescaledSurfaceCharSpeeds` to the existing options for a horizon finder
whose frame is `Frame::Distorted` and destination is
`ah::Destination::Observation`. For the binary-black-hole executable, these
are `ObservationAhA` and `ObservationAhB`. The following partial input uses
`domain::creators::CylindricalBinaryCompactObject` with its inner spherical
shells enabled, and assumes each horizon and its candidate surfaces lie inside
the corresponding inner shell. Retain the other options already configured
for each finder.

```yaml
ApparentHorizons:
  ObservationAhA:
    # Retain Criteria, InitialGuess, FastFlow, and the other finder options.
    BlocksForHorizonFind: [InnerSphereA]
    RescaledSurfaceCharSpeeds:
      ExcisionSphere: ExcisionSphereA
      NumberOfSurfaces: 10
      RelativeExcisionMargin: 1.e-7
  ObservationAhB:
    # Retain the other finder options here as well.
    BlocksForHorizonFind: [InnerSphereB]
    RescaledSurfaceCharSpeeds:
      ExcisionSphere: ExcisionSphereB
      NumberOfSurfaces: 10
      RelativeExcisionMargin: 1.e-7
```

`ExcisionSphere` must name the domain's excision sphere inside that horizon.
`NumberOfSurfaces` must be at least two. `RelativeExcisionMargin` must be finite
and positive. The values above give ten surfaces, with a small margin to keep
the innermost one outside excision. See `ah::RescaledSurfaceCharSpeedOptions`.

Omit `RescaledSurfaceCharSpeeds`, or set it to `None`, to disable
the diagnostic.
If a YAML anchor shares observation-finder options with control-system
finders, keep this option out of the shared mapping or override it with
`RescaledSurfaceCharSpeeds: None` in the control-system mappings. Enable it only
on distorted-frame observation finders.

The existing horizon-observation event and trigger set the observation times;
no additional event is required. `BlocksForHorizonFind` must cover every
candidate surface between the horizon and excision. Every selected
time-dependent block must also provide the distorted frame. For cylindrical
BBH domains, the `InnerSphereA` and `InnerSphereB` groups contain the inner
spherical shells with those maps. For `domain::creators::BinaryCompactObject`,
the corresponding wedge-shell groups are `ObjectAShell` and `ObjectBShell`;
use them only when they contain the entire candidate family and have the
required distorted-frame maps.

Do not use `All` for a BBH domain whose outer blocks lack a distorted frame.
Enabling the diagnostic sends data from every selected block, even if that
block does not intersect a candidate, so including unsupported outer blocks
is not safe. Check both frame availability and geometric coverage when
choosing the block groups.

## Surfaces and speed convention

Let $R_{\rm AH}(\Omega)$ be the horizon radius and $R_{\rm ex}(\Omega)$ the
mapped excision radius in the distorted frame, at the same angular collocation
points. For $N$ surfaces and margin $\epsilon$, the factors are

\begin{align}
q_{\min} &= (1+\epsilon)\max_\Omega
  \frac{R_{\rm ex}(\Omega)}{R_{\rm AH}(\Omega)}, \\
q_i &= 1-\left(\frac{i}{N-1}\right)^2(1-q_{\min}),
\qquad i=0,\ldots,N-1.
\end{align}

Surface $i$ has radius $q_i R_{\rm AH}(\Omega)$ about the horizon's expansion
center. Surface zero is the horizon; surface $N-1$ is closest to excision.
Quadratic spacing places more surfaces near the horizon. A valid family
requires $0<q_{\min}<1$. These are scaled horizon shapes, so the innermost one
generally differs from the actual excision shape.

On each surface the diagnostic evaluates

\begin{equation}
c(\Omega;q_i)=-\alpha+s_j\left(\beta_{\rm D}^{j}
                  +q_i\dot R_{\rm AH}\hat r^{j}\right).
\end{equation}

Here $s_j$ is the outward normal from the hole, normalized with the spatial
metric at the candidate surface, and $\hat r^j$ is the radial coordinate
direction. The shift $\beta_{\rm D}^{j}$ is the coordinate shift in the
distorted frame, including the distorted-to-inertial map velocity. The last
term accounts for the candidate's radial motion. With this excision sign
convention, positive $c$ is outflow through an inner boundary. A minimum
crossing zero indicates loss of that outflow sign somewhere on the candidate.

The recorded minimum and maximum are angular extrema of this single
characteristic branch, not extrema over all generalized-harmonic
characteristic fields. They use a frame in which the candidate surface is
stationary. Compare the result with the actual excision-boundary diagnostic,
whose shape and grid velocity can differ.

The factors are recomputed at every observation, but each speed uses the
instantaneous factor held fixed when taking the derivative. In particular,
the candidate velocity is $q_i\dot R_{\rm AH}\hat r^j$; it does not include
$\dot q_i R_{\rm AH}\hat r^j$. Thus an indexed curve does not describe the
velocity of the complete time history of that indexed surface.

The horizon and excision sphere must share a fixed expansion center, and the
grid-to-distorted map must preserve angles about that center. Moving centers
and arbitrary maps are outside this diagnostic's geometry assumptions.

## Read the output

Each completed diagnostic writes one row to the reductions file configured by
`Observers.ReductionFileName`, in the subfile
`/<HorizonMetavars>/RescaledCharSpeeds`. For example, the first hole uses
`/ObservationAhA/RescaledCharSpeeds` (an HDF5 `.dat` subfile). The columns are

```text
Time, Status,
RadiusFactor_0, MinCharSpeed_0, MaxCharSpeed_0,
RadiusFactor_1, MinCharSpeed_1, MaxCharSpeed_1, ...
```

Plot `MinCharSpeed_i` against `Time`, checking `RadiusFactor_i` to identify the
sampled surfaces at each time. A surface index is not a constant radius factor.
Use rows with `Status == 0` for physical speed comparisons. Unavailable speed
values are NaNs, so missing startup history is distinguishable from a physical
zero speed. The status values are defined by
`ah::Storage::RescaledSurfaceStatus`:

| Value | Status | Meaning |
| ---: | --- | --- |
| 0 | `Valid` | All surfaces were sampled successfully. |
| 1 | `MissingTimeDerivative` | No usable derivative, e.g. at startup. |
| 2 | `InvalidGeometry` | The geometry cannot define a valid family. |
| 3 | `OutsideDomain` | A candidate extends outside the domain. |
| 4 | `MissingBlockCoverage` | A candidate needs unselected blocks. |
| 5 | `NonfiniteSpeed` | A sampled characteristic speed is not finite. |

Sampling waits for the required volume data at the same observation time.
Missing block coverage is reported as a diagnostic status rather than waiting
for elements that were not selected to send data.

## Cost

When disabled, the finder sends no additional diagnostic fields, performs no
rescaled-surface interpolation, and retains its usual previous-horizon element
filter. When enabled, every element in the selected blocks sends its horizon
volume data and the additional lapse and distorted-frame shift. This includes
inner elements unrelated to the previous horizon, and can substantially
increase communication and retained volume data.

The diagnostic retains those data until sampling completes and interpolates
one candidate surface at a time. Increasing `NumberOfSurfaces` increases
interpolation work and the number of output columns. Start with ten surfaces
and use an observation cadence appropriate for the timescale being studied.
