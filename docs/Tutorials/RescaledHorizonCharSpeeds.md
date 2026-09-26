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

## Postprocess volume data in Python

The Python function
`spectre.Pipelines.Bbh.RescaledSurfaceCharSpeeds.rescaled_surface_char_speeds`
evaluates the same diagnostic from saved volume data. It returns a column
legend and a NumPy array, and can append the rows to an H5 Dat subfile with
the same schema as the online observation. For example:

```python
from spectre.Pipelines.Bbh.RescaledSurfaceCharSpeeds import (
    rescaled_surface_char_speeds,
)
from spectre.Strahlkorper import Frame

legend, data = rescaled_surface_char_speeds(
    "VolumeData*.h5",
    "VolumeData",
    excision_sphere="ExcisionSphereA",
    horizon_file="Surfaces.h5",
    horizon_subfile="ObservationAhA",
    horizon_frame=Frame.Inertial,
    blocks=["InnerSphereA"],
    number_of_surfaces=10,
    output_file="PostprocessedCharSpeeds.h5",
    output_subfile="ObservationAhA/RescaledCharSpeeds",
)
valid = data[:, legend.index("Status")] == 0
times = data[valid, legend.index("Time")]
min_speed = data[valid, legend.index("MinCharSpeed_5")]
```

Use the actual coefficient and volume subfile names in your files. Horizon
reduction quantities such as area or mass do not contain enough information
to reconstruct the shape. Supply either Inertial- or Distorted-frame Ylm
coefficients and set `horizon_frame` accordingly. Alternatively, supply
`horizons={time: strahlkorper, ...}` with Python Strahlkorper objects instead of
`horizon_file` and `horizon_subfile`. For a subset of volume times, pass their
observation IDs as `observation_ids=[...]`.

For saved horizons, the required volume field is the Inertial-frame
`SpacetimeMetric` (its name can be changed with `spacetime_metric_name`). The
volume files must contain the serialized domain and, for a moving domain,
the functions of time. The C++ adapter computes lapse, the Distorted-frame
coordinate shift, and inverse spatial metric at the volume nodes before
interpolating, in the same order as the online calculation. Interpolating
the spacetime metric first and subsequently deriving these fields would
produce a different discrete result.

Each requested volume time must match exactly one saved horizon within
`time_tolerance * max(1, abs(time))`. The default tolerance is `1.e-12`.
The function rejects missing or ambiguous matches and does not interpolate
horizon shapes in time. It transforms Inertial surfaces to the Distorted
frame at their own times, then re-expresses them about the fixed excision
center. A previously saved horizon can therefore have a different expansion
center without changing the diagnostic's definition of the candidate family.

The time derivative uses the current and up to two preceding saved horizons,
including horizons at times without selected volume observations. Earlier
surfaces must be transformable with the saved domain and functions of time.
If those maps do not cover the history, supply Distorted-frame surfaces or
an explicit derivative. With no earlier horizon the result has status
`MissingTimeDerivative`; a single surface does not imply zero velocity.
To supply a derivative, pass
`time_derivatives={volume_time: derivative_strahlkorper, ...}`. These
derivatives must already be in the Distorted frame about the fixed excision
center. Transforming a derivative as though it were an ordinary surface
would give an incorrect velocity.

Offline derivatives use the available saved history, so sparse horizon data
can differ from the online history. Volume resolution and output precision
also limit agreement. In particular, reduced-precision volume output need
not reproduce the original diagnostic to roundoff.

### Find horizons again from volume data

Pass an Inertial-frame initial guess instead of saved horizons to use the
existing offline FastFlow finder at each selected volume observation:

```python
from spectre.Strahlkorper import Strahlkorper

legend, data = rescaled_surface_char_speeds(
    "VolumeData*.h5",
    "VolumeData",
    excision_sphere="ExcisionSphereA",
    initial_guess=Strahlkorper[Frame.Inertial](
        l_max=16, radius=1.0, center=[4.0, 0.0, 0.0]
    ),
    horizon_data="Gh",
    blocks=["InnerSphereA"],
)
```

Choose the initial center and radius for the hole in your simulation.
`horizon_data="Gh"` requires `SpacetimeMetric`, `Pi`, and `Phi`; rename them
with `spacetime_metric_name`, `pi_name`, and `phi_name`. The helper derives
the horizon-finder fields at volume nodes and reuses the existing irregular
interpolator. The default `horizon_data="Adm"` instead requires saved
`InverseSpatialMetric`, `ExtrinsicCurvature`, and
`SpatialChristoffelSecondKind`, in addition to `SpacetimeMetric` for the
diagnostic. `horizon_tensor_names` follows the existing `find_horizon`
five-name convention; only its middle three entries are used here.

The horizon-only find skips mass, spin, and other reduction quantities, so
it does not require the spatial Ricci tensor. Each converged surface is the
initial guess for the next observation. In this mode the derivative uses
the current and up to two preceding processed volume observations, and the
first observation has missing derivative unless one is supplied explicitly.
The standard inspiral volume output contains the spacetime metric but does
not generally contain the additional fields needed to find horizons again.
Select those fields when writing the evolution output if you need this mode.

For Float-precision volume output, pass a `FastFlow` object with a tolerance
appropriate for that precision, for example:

```python
from spectre.ApparentHorizonFinder import FastFlow, FlowType

fast_flow = FastFlow(
    FlowType.Fast,
    alpha=1.0,
    beta=0.5,
    abs_tol=1.e-6,
    truncation_tol=0.01,
    divergence_tol=1.2,
    divergence_iter=5,
    max_its=300,
)
# Include fast_flow=fast_flow in rescaled_surface_char_speeds(...).
```

The workflow resets this solver before each horizon find. The default
finder's `1.e-12` absolute tolerance may be below the noise in Float output;
in that case the find can fail to converge. The example tolerance is a
starting point, and should be checked against the accuracy needed for your
diagnostic.

Both workflows require the same geometry and full spatial coverage as the
online diagnostic. `blocks` accepts block names and groups. If omitted, it
selects the blocks with the required Distorted-frame maps; explicit selection
is useful for limiting a BBH calculation to one hole's inner shell. Missing
element data or block coverage gives `MissingBlockCoverage`, with NaN speeds.
The calculation loads one volume observation at a time. Inputs are opened
read-only; output is written only when `output_file` is supplied.

For applications that already have fields in memory,
`spectre.ApparentHorizonFinder` also exposes
`sample_rescaled_surface_char_speeds`, `rescaled_surface_factors`, and
`rescaled_surface_char_speed_extrema`. These call the C++ implementation;
the Python workflow does not duplicate the speed formula or map-velocity
transformation.
