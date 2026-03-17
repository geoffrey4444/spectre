# Regge-Wheeler-Zerilli Finite-Radius Extraction Plan

## Goal
Add finite-radius Regge-Wheeler-Zerilli (RWZ) gravitational-wave extraction to
SpECTRE's GH black-hole evolutions, with the following priority order:

1. finite-radius strain `h`,
2. infrastructure needed to support `h` cleanly,
3. eventual feature parity with SpEC, including finite-radius NP scalars.

The implementation should be in a form that is:

1. reusable across single-BH and BBH executables,
2. testable in small units,
3. compatible with existing interpolation-target / observer infrastructure,
4. staged across several reviewable pull requests.

This plan assumes the first target is `spectre-container` GH evolutions
(`EvolveGhSingleBlackHole` and `EvolveGhBinaryBlackHole`), not CCE itself.
CCE already exists and should remain the asymptotic reference waveform path.

## Priority Clarification
The first user-visible milestone is finite-radius strain `h`, not finite-radius
`Psi4`. Finite-radius `Psi4` and the rest of `Psi0..Psi4` are still desirable,
but they should be planned as follow-on features unless they are needed as
enabling infrastructure for `h`.

## Definition Of Equivalent Functionality
To make sure this plan delivers the same capability as the standard SpEC BBH
RWZ path, "equivalent functionality" should mean the following concrete items,
not just "some finite-radius waveform output":

1. extraction on a family of configurable finite-radius spheres in the
   inertial frame,
2. RWZ master functions computed from GH `psi` and `kappa`,
3. the same flat-background RWZ convention that the standard SpEC BBH path
   uses,
4. complex strain `h` reconstructed from `\Phi^{(+)}`
   and `\Phi^{(-)}` with the same normalization and sign conventions used by
   current SpEC,
5. finite-radius `Psi4` available as part of the full parity target, even if it
   lands after the first strain-focused milestone,
6. auxiliary per-radius metadata needed by existing SpEC post-processing:
   `CoordRadius`, `InitialAdmEnergy`, `AverageLapse`, and `ArealRadius`,
7. a supported path to exact SpEC-compatible `GW2/*.h5` finite-radius files,
   even if the first in-core SpECTRE writer uses a native schema.

That means a strain-only output path is not sufficient for parity by itself.
The plan must also deliver the conventions, metadata, and conversion path that
let existing SpEC post-processing work unchanged.

## What SpEC Already Does
The relevant SpEC implementation is centered around
[`Evolution/AddWaveExtraction.cpp`](/home/vscode/work/builds/spec-container/Evolution/AddWaveExtraction.cpp)
and
[`Evolution/AddWaveExtraction.hpp`](/home/vscode/work/builds/spec-container/Evolution/AddWaveExtraction.hpp).

SpEC's wave-extraction stack does all of the following:

1. builds a family of extraction `Strahlkorper` surfaces at configurable radii,
2. interpolates GH metric data and derivatives to those surfaces,
3. computes finite-radius RWZ quantities from `psi` and `kappa`,
4. computes Weyl electric / magnetic tensors and Weyl characteristic fields,
5. outputs RWZ strain and finite-radius NP scalars on those surfaces,
6. writes mode data directly to H5 in a waveform-oriented format.

The most relevant SpEC ingredients are:

- RWZ computation:
  [`ComputeItems/ReggeWheelerZerilliFlat.cpp`](/home/vscode/work/builds/spec-container/ComputeItems/ReggeWheelerZerilliFlat.cpp)
  and
  [`Spectral/SpectralFunctionals/ComputeReggeWheelerZerilliFlat.hpp`](/home/vscode/work/builds/spec-container/Spectral/SpectralFunctionals/ComputeReggeWheelerZerilliFlat.hpp)
- finite-radius NP scalar extraction:
  [`ComputeItems/GrItems/WeylCharacteristicField.hpp`](/home/vscode/work/builds/spec-container/ComputeItems/GrItems/WeylCharacteristicField.hpp)
- design notes:
  [`Evolution/WeylScalarExtraction.tex`](/home/vscode/work/builds/spec-container/Evolution/WeylScalarExtraction.tex)

Two important details from SpEC that should carry over conceptually:

1. RWZ and finite-radius NP extraction are treated as surface-postprocessing on
   extraction spheres, not as volume evolution variables.
2. The extraction output is mode-oriented, not just nodal surface data.

## Exact SpEC Compatibility Target
The compatibility target should be the finite-radius waveform output produced by
the standard SpEC BBH input files:

1. [`InputFiles/Bbh/GrWaveExtraction.input`](/home/vscode/work/builds/spec-container/InputFiles/Bbh/GrWaveExtraction.input)
2. [`InputFiles/Bbh_CCE/GrWaveExtraction.input`](/home/vscode/work/builds/spec-container/InputFiles/Bbh_CCE/GrWaveExtraction.input)

Those standard inputs set:

1. `H5DatDir = __WaveDir__`, with `__WaveDir__ = GW2` in the corresponding
   `DoMultipleRuns.input`,
2. `OutputRWZh = yes`,
3. `OutputPsi4 = yes`,
4. `OutputPsi3 = no`,
5. `OutputPsi2 = no`,
6. `OutputPsi1 = no`,
7. `OutputPsi0 = no`.

So the canonical first compatibility target is the `GW2/` bundle containing at
least:

1. `rh_FiniteRadii_CodeUnits.h5`
2. `PhiPlus_FiniteRadii_CodeUnits.h5`
3. `PhiMinus_FiniteRadii_CodeUnits.h5`
4. `rPsi4_FiniteRadii_CodeUnits.h5`

From the SpEC writers in
[`Evolution/AddWaveExtraction.cpp`](/home/vscode/work/builds/spec-container/Evolution/AddWaveExtraction.cpp),
[`SurfaceFinder/StrahlkorperItems/ObserveRWZAndNPWaveforms.cpp`](/home/vscode/work/builds/spec-container/SurfaceFinder/StrahlkorperItems/ObserveRWZAndNPWaveforms.cpp),
and
[`SurfaceFinder/StrahlkorperItems/ObserveIntegralOverStrahlkorperH5Dat.cpp`](/home/vscode/work/builds/spec-container/SurfaceFinder/StrahlkorperItems/ObserveIntegralOverStrahlkorperH5Dat.cpp),
the per-file structure expected by existing tools is:

1. top-level groups named by extraction radius, e.g. `R####.dir`,
2. within each radius group:
   `CoordRadius.dat`
3. within each radius group:
   `InitialAdmEnergy.dat`
4. within each radius group:
   `AverageLapse.dat`
5. within each radius group:
   `ArealRadius.dat`
6. within each radius group:
   mode datasets named `Y_l{l}_m{m}.dat`

The waveform mode datasets store columns:

1. `time`
2. `Re[quantity]_l{l}_m{m}(R=...)`
3. `Im[quantity]_l{l}_m{m}(R=...)`

For the first strain-focused delivery in SpECTRE, exact compatibility means
matching this file naming and H5 group/dataset layout for the RWZ files, either
directly from the main extraction path or through an officially supported
conversion tool.

Decision:

1. first implementation: native SpECTRE schema plus converter,
2. required compatibility path: the converter must produce exact
   SpEC-compatible `GW2/*.h5` waveform files.

## What SpECTRE Already Has
SpECTRE already has several pieces that make this tractable:

1. GH interpolation targets on spheres:
   [`EvolveGhSingleBlackHole.hpp`](/home/vscode/work/builds/spectre-container/src/Evolution/Executables/GeneralizedHarmonic/EvolveGhSingleBlackHole.hpp)
   and
   [`EvolveGhBinaryBlackHole.hpp`](/home/vscode/work/builds/spectre-container/src/Evolution/Executables/GeneralizedHarmonic/EvolveGhBinaryBlackHole.hpp)
2. surface callbacks:
   [`ObserveSurfaceData.hpp`](/home/vscode/work/builds/spectre-container/src/ParallelAlgorithms/Interpolation/Callbacks/ObserveSurfaceData.hpp)
   and
   [`ObserveTimeSeriesOnSurface.hpp`](/home/vscode/work/builds/spectre-container/src/ParallelAlgorithms/Interpolation/Callbacks/ObserveTimeSeriesOnSurface.hpp)
3. GH-side 3+1 compute tags for the ingredients of finite-radius wave
   extraction:
   [`GeneralizedHarmonicBase.hpp`](/home/vscode/work/builds/spectre-container/src/Evolution/Executables/GeneralizedHarmonic/GeneralizedHarmonicBase.hpp),
   [`WeylElectric.hpp`](/home/vscode/work/builds/spectre-container/src/PointwiseFunctions/GeneralRelativity/WeylElectric.hpp),
   [`WeylMagnetic.hpp`](/home/vscode/work/builds/spectre-container/src/PointwiseFunctions/GeneralRelativity/WeylMagnetic.hpp),
   [`ExtrinsicCurvature.hpp`](/home/vscode/work/builds/spectre-container/src/PointwiseFunctions/GeneralRelativity/ExtrinsicCurvature.hpp)
4. a complex `Psi4` pointwise function already exists:
   [`Psi4.hpp`](/home/vscode/work/builds/spectre-container/src/PointwiseFunctions/GeneralRelativity/Psi4.hpp)
   and
   [`Psi4.cpp`](/home/vscode/work/builds/spectre-container/src/PointwiseFunctions/GeneralRelativity/Psi4.cpp)
5. spin-weighted harmonic infrastructure already exists in the CCE module,
   including Goldberg-mode output machinery:
   [`ScriObserveInterpolated.hpp`](/home/vscode/work/builds/spectre-container/src/Evolution/Systems/Cce/Actions/ScriObserveInterpolated.hpp)

This means the missing work is not "build everything from scratch". The main
gap is that finite-radius extraction is not yet wired together as a GH surface
feature.

## Missing Pieces In SpECTRE
Compared to SpEC, the main missing functionality is:

1. a dedicated finite-radius wave-extraction interpolation target for GH,
2. configurable extraction radii / `LMax` / output controls in GH executables,
3. compute tags on the target for complex finite-radius NP quantities, rather
   than only volume `Psi4Real`,
4. a reusable mode-output callback for complex spin-weighted data on a sphere,
5. RWZ master-function computation on extraction spheres using SpEC's current
   flat-background convention,
6. strain reconstruction with SpEC-compatible normalization and sign choices,
7. per-radius metadata output for `CoordRadius`, `InitialAdmEnergy`,
   `AverageLapse`, and `ArealRadius`,
8. a supported converter from native SpECTRE output to exact SpEC `GW2/*.h5`
   files,
9. end-to-end tests and YAML coverage for the new feature.

Secondary gaps:

1. there is no obvious finite-radius waveform H5 schema yet for RWZ output,
2. there is not yet a dedicated place in SpECTRE for RWZ-on-spheres utilities
   that are shared across executables and tools,
3. there is not yet a clear answer on whether we should expose only RWZ plus
   complex `Psi4`, or the full `Psi0..Psi4` family.

Given the stated priority, the recommended first answer is:

1. expose RWZ and strain `h` first,
2. defer the full `Psi0..Psi4` family.

## Recommended Design Direction
The cleanest SpECTRE-native design is:

1. add a new GH interpolation target family for extraction spheres in the
   inertial frame,
2. compute RWZ quantities on the target with `compute_items_on_target`,
3. compute the auxiliary surface metadata needed by the SpEC finite-radius
   files on the same targets,
4. reconstruct strain `h` from the RWZ master functions on those targets,
5. add a waveform-oriented callback for native mode output,
6. add a Python/CLI conversion tool that converts the native output into the
   exact SpEC `GW2/*.h5` contract,
7. reuse existing surface callbacks for optional nodal debug output,
8. treat finite-radius NP-scalar output as a later extension that can reuse the
   same extraction target.

This keeps the executable wiring thin and makes later extensions possible:

- ringdown-only workflows,
- finite-radius `Psi4` mode extraction without RWZ,
- comparison tools against CCE,
- potential future support for other systems that interpolate GH fields to
  spheres.

## Proposed PR Sequence

### PR 1: Finite-Radius Extraction Target Skeleton
Scope:

1. add new GH interpolation-target option tags for extraction spheres,
2. add a new target for single-BH and BBH executables,
3. support configurable radii, center, angular ordering, and `LMax`,
4. ensure the target is explicitly inertial-frame based so it can mirror the
   standard SpEC path,
5. initially output only nodal surface diagnostics already easy to compute.

Likely files:

- GH executable headers under
  `src/Evolution/Executables/GeneralizedHarmonic/`
- shared option / metavars helpers if a common target helper is worthwhile

Why first:

1. it establishes the user-facing hook,
2. it exercises interpolation / observation plumbing early,
3. it can land before any RWZ math.

Suggested tests:

1. ActionTesting / interpolation-target tests for the new target,
2. input-file parse test with one extraction sphere,
3. execute_check_output smoke test that writes a surface file.

### PR 2: RWZ Algorithm Library
Scope:

1. port or rederive the flat-background RWZ extraction used by SpEC,
2. place it in a reusable library location, not executable code,
3. define clear inputs:
   metric perturbation pieces on a sphere,
   time derivative information,
   radial derivatives / normal information,
   areal or coordinate radius conventions,
4. define the corresponding strain `h` reconstruction path from RWZ outputs,
5. document and test the exact normalization and sign conventions that must
   agree with the standard SpEC BBH path.

Recommendation:

1. preserve SpEC's current flat-background RWZ definition first,
2. include the strain output needed by users in the same conceptual feature
   slice, even if some storage/output wiring lands in the next PR,
3. do not mix in alternate RWZ conventions or Kerr/general-background
   generalizations in the first implementation.

Likely files:

- new files under
  `src/PointwiseFunctions/GeneralRelativity/Surfaces/`
  or a nearby wave-extraction namespace
- tests under `tests/Unit/PointwiseFunctions/GeneralRelativity/Surfaces/`

Why separate:

1. the math and conventions need focused review,
2. this is the highest-priority waveform ingredient.

Suggested tests:

1. unit tests for low-order perturbative data,
2. comparison against SpEC-generated reference data on a fixed sphere,
3. checks of parity splitting and zeroing for `l < 2`,
4. direct tests of strain reconstruction from the RWZ master functions.

### PR 3: Surface Metadata And Native Output Callback
Scope:

1. add a new interpolation callback that writes waveform-oriented mode time
   series,
2. support at least RWZ master functions and strain `h`,
3. output the per-radius metadata needed to reproduce SpEC's finite-radius
   files later:
   `CoordRadius`,
   `InitialAdmEnergy`,
   `AverageLapse`,
   `ArealRadius`,
4. define a stable native H5 layout for finite-radius extraction output.

Recommendation for first version:

1. write one reduction dataset per quantity per extraction radius,
2. store time followed by mode data in a layout appropriate for the quantity,
3. store enough metadata that the converter can reproduce SpEC's `R####.dir`
   groups without guesswork,
4. keep the native layout simple and close to the existing CCE conventions
   where practical.

Follow-up requirement:

1. if this first callback does not already write exact SpEC-compatible H5
   waveform files, add a supported conversion path in SpECTRE's Python/CLI
   tooling that converts the native output into H5 files matching SpEC's
   finite-radius RWZ output format exactly.

Chosen direction:

1. this PR sequence should assume the first callback writes a native
   SpECTRE-oriented schema,
2. the conversion tool is therefore part of the planned supported workflow, not
   an optional convenience script.

Likely files:

- new callback under
  `src/ParallelAlgorithms/Interpolation/Callbacks/`
- maybe small shared helpers extracted from CCE output code

Why separate:

1. it is reusable for both RWZ/strain now and NP scalars later,
2. it isolates H5 schema discussions from GH executable plumbing.

Suggested tests:

1. unit test for callback legend/data ordering,
2. test for known mock RWZ / strain mode rows,
3. test for metadata rows for `CoordRadius`, `InitialAdmEnergy`,
   `AverageLapse`, and `ArealRadius`.

### PR 4: Native Metadata Completion And Stable RWZ Schema
Scope:

1. finish the native waveform callback so it writes all metadata needed for
   exact SpEC conversion:
   `CoordRadius`,
   `InitialAdmEnergy`,
   `AverageLapse`,
   `ArealRadius`,
2. carry `InitialAdmEnergy` through the GH user-facing options explicitly,
   since SpEC's BBH path supplies it as an input metadata value rather than
   deriving it from the extraction sphere,
3. lock a stable native H5 schema for RWZ `h`, `\Phi^{(+)}`, and
   `\Phi^{(-)}` that the converter can target without further C++ churn,
4. add focused tests for metadata values, legends, and per-radius grouping in
   the native output.

Why here:

1. it finishes the minimal native contract before any compatibility tooling is
   layered on top,
2. it keeps the converter PR narrow and mostly Python/H5 focused,
3. it makes RWZ `h` validation against SpEC possible before any NP-scalar work.

Suggested tests:

1. unit test for metadata rows and legends,
2. regression test that multiple extraction radii produce separate native
   groups with the expected scalar metadata,
3. smoke test that `InitialAdmEnergy` survives from input options to output.

### PR 5: Converter To Exact SpEC `GW2` RWZ Files
Scope:

1. add a Python/CLI tool in SpECTRE that reads the native RWZ extraction
   output,
2. write exact SpEC-compatible files:
   `rh_FiniteRadii_CodeUnits.h5`,
   `PhiPlus_FiniteRadii_CodeUnits.h5`,
   `PhiMinus_FiniteRadii_CodeUnits.h5`,
3. reproduce the expected `R####.dir` layout and datasets:
   `CoordRadius.dat`,
   `InitialAdmEnergy.dat`,
   `AverageLapse.dat`,
   `ArealRadius.dat`,
   `Y_l{l}_m{m}.dat`,
4. validate the converter against the standard SpEC BBH file structure and
   legends.

Likely files:

- Python CLI entry points under `support/Python/` and `src/IO/H5/Python/`
- Python tests under `tests/Unit/IO/H5/Python/`

Recommendation:

1. treat the converter as part of the supported first-delivery workflow, not
   as an optional helper,
2. target exact compatibility with the `GW2/*.h5` files produced by the
   standard SpEC BBH inputs,
3. keep the converter schema-focused and limited to RWZ `h`,
   `\Phi^{(+)}`, and `\Phi^{(-)}` first.

Suggested tests:

1. unit tests for converter naming and group layout,
2. golden-data test comparing converter output structure to a known-good SpEC
   layout,
3. regression test that a native extraction file with multiple radii produces
   the expected `GW2/*.h5` outputs.

### PR 6: RWZ `h` Validation Against SpEC And Analytic Data
Scope:

1. add regression-style validation scripts or tests for RWZ strain `h`,
2. compare SpECTRE finite-radius RWZ `h` to SpEC reference output on matched
   snapshots and, where practical, matched short evolutions,
3. add analytic-data validation for the RWZ library and extraction path using a
   known outgoing perturbative waveform or similarly controlled solution,
4. use these checks to lock the sign, normalization, and metadata conventions
   before extending the feature to NP scalars.

Why here:

1. the highest-priority scientific risk is that RWZ `h` match SpEC before we
   broaden the feature set,
2. validation now reduces the chance of building NP-scalar work on top of a
   convention mismatch,
3. this mirrors the user priority: trust RWZ `h` first, extend later.

Suggested tests:

1. direct SpECTRE-vs-SpEC comparison of converted `rh_FiniteRadii_CodeUnits.h5`
   mode data for a representative BBH or single-BH case,
2. analytic waveform test with known outgoing content and expected
   `\ell,m` structure,
3. regression checks that flat-space and pure-gauge data produce zero RWZ
   signal,
4. lightweight documentation on what is being validated and what tolerances are
   expected.

### PR 7: Finite-Radius `Psi4`
Scope:

1. promote complex finite-radius `Psi4` to a first-class extraction-sphere
   output,
2. add a reusable compute-tag path for `Psi4` on the extraction target,
3. if useful, add mode output using existing `Swsh` / Goldberg infrastructure,
4. extend the converter so the standard SpEC-compatible path also writes
   `rPsi4_FiniteRadii_CodeUnits.h5`.

Why after RWZ validation:

1. `h` is the higher priority deliverable,
2. this keeps the NP-scalar work from obscuring the RWZ convention checks,
3. the `Psi4` path can reuse the validated extraction target and callback
   infrastructure.

### PR 8: Remaining NP Scalars And Full SpEC Parity
Scope:

1. extend finite-radius extraction to `Psi3`, `Psi2`, `Psi1`, and `Psi0`,
2. add any remaining SpEC-style extraction products still missing,
3. close known parity gaps in output controls and analysis convenience,
4. if the main extraction path still writes a native SpECTRE schema, ensure the
   supported conversion tool produces exact SpEC-compatible finite-radius RWZ
   files.

This PR is explicitly a parity milestone, not a first-delivery requirement.

## Recommended Scope Cuts For The First Implementation
To keep the first end-to-end version reviewable, I recommend:

1. support GH single-BH and BBH only,
2. support fixed extraction center from input options first,
3. support RWZ plus strain `h` first, but include the metadata and converter
   needed for immediate SpEC-tool compatibility,
4. treat complex `Psi4` as a follow-up unless it materially simplifies testing
   or validation,
5. reuse existing CCE harmonic transforms where they help, rather than invent a
   second transform stack,
6. do not cut the flat-background convention work, sign-convention lock-in, or
   metadata output, because those are part of functional equivalence rather
   than optional polish.

## Risks And Decisions To Resolve Early

### 1. RWZ Convention Lock-In
SpEC's RWZ implementation has convention choices and historical sign updates.
Before PR 6 starts, we should explicitly lock:

1. normalization,
2. sign conventions,
3. radius convention,
4. H5 dataset naming.

### 2. Choice Of Harmonic Basis For RWZ / Strain Output
There are at least two plausible paths:

1. write RWZ / strain modes directly in the natural basis of the master
   functions,
2. force everything through the same spin-weighted infrastructure.

Recommendation:

1. use the simplest basis natural to RWZ / strain first,
2. reuse existing `Swsh` machinery later for NP scalars.

### 3. Output Schema
If the output layout diverges too far from CCE or too far from SpEC, downstream
analysis becomes harder.

Recommendation:

1. the ultimate target should be exact SpEC-compatible finite-radius RWZ H5
   output so existing post-processing tools can be reused unchanged,
2. a first implementation may use a simpler SpECTRE-native layout only if
   SpECTRE also provides a supported conversion tool that writes H5 files
   matching the standard SpEC `GW2/` finite-radius RWZ output format exactly,
3. document any temporary schema differences clearly and treat them as
   transitional rather than permanent.

The native schema should also preserve enough information to reconstruct the
exact SpEC output layout without inference or lossy renaming.

### 4. Where To Compute Surface Quantities
Some pieces can be computed on the volume before interpolation, others on the
target after interpolation.

Recommendation:

1. interpolate the GH primitive/source variables to the sphere,
2. compute RWZ and strain on the target,
3. avoid interpolating a large set of already-derived volume tensors unless
   profiling later shows a clear need.

That keeps frame / tetrad / surface-geometry choices local to the extraction
target.

## Concrete Near-Term Worklist
If someone starts implementation immediately, the first coding pass should be:

1. prototype a new GH extraction-sphere interpolation target,
2. prototype the RWZ library and strain reconstruction on that target,
3. prototype native output for RWZ / strain plus the required metadata,
4. prototype the Python/CLI conversion tool interface in parallel,
5. verify that the native schema can reproduce exact SpEC `GW2` files,
6. only then add finite-radius `Psi4` on top.

This order reduces risk because it proves the end-to-end data path before the
full parity surface feature set lands.

## Linear Walkthrough Of The Standard SpEC BBH RWZ Path

This section follows only the RWZ path used by the standard SpEC BBH input
files. It starts from the input-file entry point and then walks through the
code in the order it is used.

### Step 1: The Standard `Bbh` Input Files Enable RWZ Extraction
The standard BBH include chain starts in
[`GrDataBoxItems.input`](/home/vscode/work/builds/spec-container/InputFiles/Bbh/GrDataBoxItems.input),
which reads
[`GrWaveExtraction.input`](/home/vscode/work/builds/spec-container/InputFiles/Bbh/GrWaveExtraction.input).

That file instantiates `AddWaveExtraction(...)` with:

1. `OutputRWZh = yes`,
2. `OutputPsi4 = yes`,
3. `OutputPsi3 = no`,
4. `OutputPsi2 = no`,
5. `OutputPsi1 = no`,
6. `OutputPsi0 = no`,
7. `OutputCceScalars = yes`.

So in the standard BBH path, finite-radius RWZ strain is on by default,
finite-radius `Psi4` is also on, and the lower NP scalars are not part of the
default production path.

The output subdirectory is `GW2`, because
[`DoMultipleRuns.input`](/home/vscode/work/builds/spec-container/InputFiles/Bbh/DoMultipleRuns.input)
sets `__WaveDir__ = GW2`.

### Step 2: `AddWaveExtraction` Builds The Extraction Spheres
In
[`AddWaveExtraction.cpp`](/home/vscode/work/builds/spec-container/Evolution/AddWaveExtraction.cpp),
`AddWaveExtraction::ComputeOptionsToStrahlkorperDataBoxes` constructs a family
of extraction `Strahlkorper` DataBoxes with:

1. `OutputBaseName = WaveExtraction`,
2. `OutputNames = <<FromRadii>>`,
3. `WhichTensors = psi,kappa` plus any requested Weyl characteristic fields,
4. `Strahlkorpers = MappedSpheres(...)` centered at the origin,
5. `L_mesh = 16`.

So the SpEC RWZ path is organized as a set of extraction spheres, one per
radius, each holding the interpolated GH data and the surface-local quantities
used for wave extraction.

### Step 3: SpEC Populates The Extraction-Sphere DataBoxes
Still in
[`AddWaveExtraction.cpp`](/home/vscode/work/builds/spec-container/Evolution/AddWaveExtraction.cpp),
`AddWaveExtraction::AddDataBoxItems` adds the compute items needed by the
standard BBH extraction path.

For RWZ, the important pieces are:

1. `psi` and `kappa` from GH,
2. `Add3Plus1ItemsFromGhPsiKappa(...)`,
3. `EvaluateScalarFormula(Output = One; Formula = 1; ...)`.

For finite-radius `Psi4`, SpEC also adds:

1. `WeylElectric`,
2. `WeylMagnetic`,
3. `WeylCharacteristicField` for `U8+`.

So the standard BBH path computes RWZ from `psi` and `kappa`, while the
finite-radius `Psi4` path is built from the electric and magnetic Weyl tensors.
These are parallel extraction products sharing the same extraction spheres.

### Step 4: SpEC Installs The Waveform Writers
In
[`AddWaveExtraction.cpp`](/home/vscode/work/builds/spec-container/Evolution/AddWaveExtraction.cpp),
`AddWaveExtraction::AddObservers` installs the two observers that matter for the
standard finite-radius RWZ files:

1. `RWZAndNPWaveforms(...)`,
2. `IntegralOverStrahlkorperH5Dat(...)`.

The standard BBH observer configuration writes:

1. `rh_FiniteRadii_CodeUnits.h5`,
2. `PhiPlus_FiniteRadii_CodeUnits.h5`,
3. `PhiMinus_FiniteRadii_CodeUnits.h5`,
4. `rPsi4_FiniteRadii_CodeUnits.h5`.

It also requests:

1. `AverageLapse`,
2. `ArealRadius`.

Those auxiliary datasets are appended into the same waveform files, which is
why existing SpEC post-processing expects them there.

### Step 5: At Observation Time, `RWZAndNPWaveforms` Fetches Every Surface
In
[`ObserveRWZAndNPWaveforms.cpp`](/home/vscode/work/builds/spec-container/SurfaceFinder/StrahlkorperItems/ObserveRWZAndNPWaveforms.cpp),
`RWZAndNPWaveforms::Observe`:

1. retrieves the list of `WaveExtraction` surface DataBoxes,
2. triggers the interpolation-backed data access for each one,
3. computes the mean coordinate radius from the `Strahlkorper` coefficients,
4. dispatches per-surface work for RWZ and any requested NP scalars.

The coordinate radius written to file is effectively the monopole part of the
surface shape, $r = c_{00} / \sqrt{8}$ in SpEC's normalization.

### Step 6: The RWZ Calculation Calls `ComputeReggeWheelerZerilliFlat`
For each extraction sphere,
`RWZAndNPWaveforms::ComputeRWZExtraction` calls
[`ComputeReggeWheelerZerilliFlat.cpp`](/home/vscode/work/builds/spec-container/Spectral/SpectralFunctionals/ComputeReggeWheelerZerilliFlat.cpp).

This is the key background choice:

$$
\text{The standard SpEC BBH RWZ path perturbs about flat space, not
Schwarzschild.}
$$

That is explicit in
[`ReggeWheelerZerilli.tex`](/home/vscode/work/builds/spec-container/Spectral/SpectralFunctionals/ReggeWheelerZerilli.tex),
which says "Here we perturb about flat space" and uses the background line
element

$$
ds^2 = -dt^2 + dr^2 + r^2(d\theta^2 + \sin^2\theta\, d\phi^2).
$$

So the production BBH RWZ path is a flat-background RWZ extraction on finite
radius spheres in inertial coordinates.

### Step 7: SpEC Forms Metric Perturbations Relative To Minkowski
The documentation in
[`ReggeWheelerZerilli.tex`](/home/vscode/work/builds/spec-container/Spectral/SpectralFunctionals/ReggeWheelerZerilli.tex)
and the code in
[`ComputeReggeWheelerZerilliFlat.cpp`](/home/vscode/work/builds/spec-container/Spectral/SpectralFunctionals/ComputeReggeWheelerZerilliFlat.cpp)
agree on the starting point:

$$
\delta g_{\alpha\beta} = \psi_{\alpha\beta} - \eta_{\alpha\beta},
$$

$$
\partial_t \delta g_{\alpha\beta} = -\Pi_{\alpha\beta}
= -\kappa_{0\alpha\beta},
$$

$$
\partial_r \delta g_{\alpha\beta} =
n^i \Phi_{i\alpha\beta} = n^i \kappa_{i\alpha\beta}.
$$

In code, `ComputeReggeWheelerZerilliFlatWork1` constructs:

1. the spatial tensor $T_{ij} = \delta g_{ij}$,
2. its time derivative $\dot{T}_{ij}$,
3. its radial derivative $T'_{ij}$,
4. the spatial vector $v_i = \delta g_{ti}$,
5. its radial derivative $v'_i$.

The normal used for the radial derivative is normalized with the flat metric.

### Step 8: SpEC Interpolates To Angular Points Fixed In The Inertial Frame
If the grid-to-inertial map is nontrivial, SpEC does not simply decompose the
surface data at grid-frame angular coordinates. In
`ComputeReggeWheelerZerilliFlatWork4`, it:

1. chooses angular collocation points $(\bar{\theta}, \bar{\phi})$ fixed in the
   inertial frame,
2. builds inertial Cartesian points on a sphere of constant inertial radius
   $r$,
3. inverse-maps those points back to the grid frame,
4. interpolates $T$, $\dot{T}$, $T'$, $v$, and $v'$ to those points.

This makes the subsequent harmonic decomposition live on an inertial-frame
sphere, even though the interpolation itself is performed in the grid frame.

### Step 9: SpEC Decomposes The Perturbations Into Harmonic Amplitudes
Next, `ComputeReggeWheelerZerilliFlatWork2` and
`ComputeReggeWheelerZerilliFlatWork3` decompose the tensor and vector data into
the amplitudes used in the RWZ formalism.

Defining $\lambda = (\ell - 1)(\ell + 2)$, the flat-background RWZ scalars used
by SpEC are:

$$
\Phi^{(-)} =
\frac{r}{\lambda}
\left(\dot{h}_r - h_t' + \frac{2}{r} h_t\right),
$$

$$
\Phi^{(+)} =
\frac{r}{\lambda \ell(\ell+1)}
\left(2 Z_r + \lambda K^{(\mathrm{inv})}\right).
$$

The code computes exactly these mode-by-mode combinations from the harmonic
amplitudes of the metric perturbation and its derivatives.

### Step 10: Modes With $\ell \le 1$ Are Not Used For RWZ Radiation
Back in
[`ObserveRWZAndNPWaveforms.cpp`](/home/vscode/work/builds/spec-container/SurfaceFinder/StrahlkorperItems/ObserveRWZAndNPWaveforms.cpp),
after `ComputeReggeWheelerZerilliFlat(...)` returns, SpEC explicitly zeros
coefficients that are too high for the current surface mesh and skips
nonradiative low multipoles in output when `SkipUnusedL = yes`.

Conceptually, RWZ radiation is only used for $\ell \ge 2$, consistent with the
formalism and the documentation.

### Step 11: SpEC Reconstructs The Complex Strain From $\Phi^{(+)}$ And $\Phi^{(-)}$
The documentation gives the leading-order strain reconstruction as

$$
h = \frac{1}{r}\sum_{\ell m}
\sqrt{\ell(\ell+1)\lambda}\,
\left(\Phi^{(+)}_{\ell m} + i \Phi^{(-)}_{\ell m}\right)\,
{}_{-2}Y^{\ell m}.
$$

In the observer code, SpEC stores mode coefficients of `rh`, so the factor of
$r^{-1}$ is absorbed and the stored quantity is effectively

$$
r h_{\ell m} =
\sqrt{(\ell-1)\ell(\ell+1)(\ell+2)}\,
\left(\Phi^{(+)}_{\ell m} + i \Phi^{(-)}_{\ell m}\right).
$$

The implementation in
[`ObserveRWZAndNPWaveforms.cpp`](/home/vscode/work/builds/spec-container/SurfaceFinder/StrahlkorperItems/ObserveRWZAndNPWaveforms.cpp)
assembles this as

$$
\mathrm{Re}[r h_{\ell m}] =
\sqrt{(\ell-1)\ell(\ell+1)(\ell+2)}
\left(\mathrm{Re}\,\Phi^{(+)}_{\ell m}
- \mathrm{Im}\,\Phi^{(-)}_{\ell m}\right),
$$

$$
\mathrm{Im}[r h_{\ell m}] =
\sqrt{(\ell-1)\ell(\ell+1)(\ell+2)}
\left(\mathrm{Im}\,\Phi^{(+)}_{\ell m}
+ \mathrm{Re}\,\Phi^{(-)}_{\ell m}\right).
$$

This is the current convention used by the standard BBH path.

### Step 12: SpEC Writes The RWZ Files In The `GW2/*.h5` Layout
`RWZAndNPWaveforms::OutputRWZExtraction` writes:

1. `rh_FiniteRadii_CodeUnits.h5`,
2. `PhiPlus_FiniteRadii_CodeUnits.h5`,
3. `PhiMinus_FiniteRadii_CodeUnits.h5`.

For each extraction radius it creates a group like `R####.dir` and writes:

1. `CoordRadius.dat`,
2. `InitialAdmEnergy.dat`,
3. `Y_l{l}_m{m}.dat` for each mode.

Each mode dataset has columns:

1. `time`,
2. `Re[quantity]_l{l}_m{m}(R=...)`,
3. `Im[quantity]_l{l}_m{m}(R=...)`.

This is the concrete layout that downstream SpEC tools expect.

### Step 13: SpEC Appends `AverageLapse` And `ArealRadius` Into Those Same Files
The second observer,
[`ObserveIntegralOverStrahlkorperH5Dat.cpp`](/home/vscode/work/builds/spec-container/SurfaceFinder/StrahlkorperItems/ObserveIntegralOverStrahlkorperH5Dat.cpp),
computes surface integrals on the same extraction spheres.

In the standard BBH configuration, `AddWaveExtraction::AddObservers` requests:

1. `AverageLapse = A/B`,
2. `ArealRadius = \sqrt{B/(4\pi)}`,

where $A$ is the integral of `Lapse` over the surface with the physical area
element and $B$ is the integral of `One` over the same surface, so $B$ is the
surface area.

Those values are written into the same `R####.dir` groups as:

1. `AverageLapse.dat`,
2. `ArealRadius.dat`.

That is why the standard SpEC waveform files contain both mode data and these
surface-level metadata time series.

### Step 14: Finite-Radius `Psi4` Runs Alongside RWZ, But It Is A Separate Calculation
The same observer also outputs finite-radius `Psi4`, but that path is not used
to construct `rh`. Instead, it uses the Weyl characteristic field $U^{8+}$ and
the Newman-Penrose contraction described in
[`WeylScalarExtraction.tex`](/home/vscode/work/builds/spec-container/Evolution/WeylScalarExtraction.tex):

$$
\Psi_4 = U^{8+}_{ij}\,\bar{m}^i \bar{m}^j.
$$

The documentation also states SpEC's asymptotic convention:

$$
\Psi_4 = -\ddot{h}
\quad \text{in the asymptotic limit.}
$$

So, in the standard BBH path, RWZ strain and finite-radius `Psi4` are two
separate extractions written side by side, not one derived from the other.

### Documentation Summary And Literature Trail
The most relevant SpEC documentation for this path is:

1. [`ReggeWheelerZerilli.tex`](/home/vscode/work/builds/spec-container/Spectral/SpectralFunctionals/ReggeWheelerZerilli.tex),
2. [`WeylScalarExtraction.tex`](/home/vscode/work/builds/spec-container/Evolution/WeylScalarExtraction.tex),
3. [`SignChangeInRWZExtraction.tex`](/home/vscode/work/builds/spec-container/Evolution/SignChangeInRWZExtraction.tex).

What those notes say, in short:

1. SpEC's RWZ extraction is a gauge-invariant perturbative extraction about a
   flat Minkowski background, not a Schwarzschild background.
2. The code forms perturbations from the GH spacetime metric and its
   derivatives, decomposes them into tensor harmonics on extraction spheres,
   builds the gauge-invariant RWZ master functions, and reconstructs the strain
   from them.
3. The Weyl-scalar extraction is a separate surface-postprocessing path using
   the electric and magnetic Weyl tensors and characteristic fields.
4. SpEC's current sign convention is chosen so that asymptotically
   $\Psi_4 = -\ddot{h}$.
5. There was an explicit historical sign change in `PhiPlus` and in `h`, and
   SpEC records those changes in `VersionHist.ver` for the affected files.

The literature references called out directly in the SpEC documentation for the
RWZ path include:

1. Sarbach and Tiglio (2001) for the gauge-invariant perturbation formalism,
2. Nagar and Rezzolla (2005) and Ruiz, Takahashi, Alcubierre, and Nunez (2008)
   for the strain reconstruction formula,
3. Rinne, Buchman, and Scheel in the extraction notes that SpEC cites and
   builds on,
4. Buchman and Sarbach (2007) in the RWZ notes,
5. Newman-Penrose and Weyl-scalar sign-convention references collected in
   `WeylScalarExtraction.tex`.

### Bottom Line For SpECTRE Planning
If SpECTRE wants to match the standard SpEC BBH RWZ path, then the closest
target is:

1. finite-radius extraction on inertial-frame spheres,
2. flat-background RWZ, not Schwarzschild-background RWZ,
3. strain reconstructed from $\Phi^{(+)}$ and $\Phi^{(-)}$ with SpEC's current
   sign convention,
4. output eventually convertible to the exact `GW2/*.h5` layout used by SpEC.
## Progress Log

### Completed Implementation Work

#### PR 1
Commit: `b911057208`

Summary:

1. added the `FiniteRadiusExtraction` interpolation target skeleton to the GH
   single-BH and BBH executables,
2. added user-facing input-file coverage for the new target,
3. verified the target parses and participates in the executable plumbing.

Files:

1. `src/Evolution/Executables/GeneralizedHarmonic/EvolveGhSingleBlackHole.hpp`
2. `src/Evolution/Executables/GeneralizedHarmonic/EvolveGhBinaryBlackHole.hpp`
3. `tests/InputFiles/GeneralizedHarmonic/KerrSchild.yaml`
4. `tests/InputFiles/GeneralizedHarmonic/CylindricalBinaryBlackHole.yaml`

Verification:

1. built `EvolveGhSingleBlackHole` and `EvolveGhBinaryBlackHole`,
2. ran the GH input-file parse / smoke tests for the updated YAML files.

#### PR 2
Commit: `1209789fdf`

Summary:

1. added the flat-background RWZ master-function library,
2. kept the background and conventions aligned with the standard SpEC BBH path,
3. added unit tests for RWZ master functions and strain reconstruction.

Files:

1. `src/PointwiseFunctions/GeneralRelativity/Surfaces/ReggeWheelerZerilli.hpp`
2. `src/PointwiseFunctions/GeneralRelativity/Surfaces/ReggeWheelerZerilli.cpp`
3. `tests/Unit/PointwiseFunctions/GeneralRelativity/Surfaces/Test_ReggeWheelerZerilli.cpp`

Verification:

1. built `Test_GrSurfaces`,
2. ran `/home/vscode/work/builds/build/bin/Test_GrSurfaces`.

#### PR 3
Commit: `01514632e5`

Summary:

1. wired GH finite-radius RWZ extraction and native waveform output,
2. added the `ObserveReggeWheelerZerilli` callback,
3. preserved the tensor-harmonic decomposition route rather than switching to
   a scalar-harmonic rewrite that had not yet been validated against SpEC.

Files:

1. `src/ParallelAlgorithms/Interpolation/Callbacks/ObserveReggeWheelerZerilli.hpp`
2. `src/Evolution/Executables/GeneralizedHarmonic/EvolveGhSingleBlackHole.hpp`
3. `src/Evolution/Executables/GeneralizedHarmonic/EvolveGhBinaryBlackHole.hpp`
4. `src/PointwiseFunctions/GeneralRelativity/Surfaces/ReggeWheelerZerilli.hpp`
5. `src/PointwiseFunctions/GeneralRelativity/Surfaces/ReggeWheelerZerilli.cpp`

Verification:

1. rebuilt the GH executables,
2. reran `/home/vscode/work/builds/build/bin/Test_GrSurfaces`,
3. reran the GH input-file parse / smoke tests.

#### PR 4
Commit: `9850437e04`

Summary:

1. completed the native metadata path for `CoordRadius`, `InitialAdmEnergy`,
   `AverageLapse`, and `ArealRadius`,
2. added `InitialAdmEnergy` as a GH user-facing option carried through to the
   output callback,
3. computed `AverageLapse` and `ArealRadius` directly from the extraction
   sphere data needed for SpEC-compatible post-processing later.

Files:

1. `src/PointwiseFunctions/GeneralRelativity/Surfaces/ReggeWheelerZerilli.hpp`
2. `src/PointwiseFunctions/GeneralRelativity/Surfaces/ReggeWheelerZerilli.cpp`
3. `src/ParallelAlgorithms/Interpolation/Callbacks/ObserveReggeWheelerZerilli.hpp`
4. `src/Evolution/Executables/GeneralizedHarmonic/EvolveGhSingleBlackHole.hpp`
5. `src/Evolution/Executables/GeneralizedHarmonic/EvolveGhBinaryBlackHole.hpp`
6. `tests/Unit/PointwiseFunctions/GeneralRelativity/Surfaces/Test_ReggeWheelerZerilli.cpp`
7. `tests/InputFiles/GeneralizedHarmonic/KerrSchild.yaml`
8. `tests/InputFiles/GeneralizedHarmonic/CylindricalBinaryBlackHole.yaml`

Verification:

1. rebuilt `Test_GrSurfaces`,
2. reran `/home/vscode/work/builds/build/bin/Test_GrSurfaces`,
3. rebuilt the GH executables,
4. reran the GH input-file parse / smoke tests.

#### PR 5
Commit: `fbdddfea09`

Summary:

1. added a Python / CLI converter from the native RWZ output to exact
   SpEC-compatible `GW2/*.h5` files for `rh`, `PhiPlus`, and `PhiMinus`,
2. targeted the standard SpEC BBH finite-radius layout with `R####.dir`,
   metadata dat files, and `Y_l{l}_m{m}.dat` datasets,
3. added Python tests for the conversion path.

Files:

1. `src/IO/H5/Python/ConvertFiniteRadiusRwzToSpec.py`
2. `support/Python/__main__.py`
3. `src/IO/H5/Python/CMakeLists.txt`
4. `tests/Unit/IO/H5/Python/Test_ConvertFiniteRadiusRwzToSpec.py`
5. `tests/Unit/IO/H5/Python/CMakeLists.txt`

Verification:

1. ran the converter unit test with the build-tree `PYTHONPATH`,
2. checked that the emitted files matched the expected SpEC naming and H5
   structure.

#### PR 6
Status: started in this container

Summary:

1. added a Python comparison utility for SpEC-style finite-radius RWZ files so
   validation can compare converted SpECTRE output directly to SpEC output,
2. wired the comparison utility into the SpECTRE CLI and Python unit tests,
3. deferred the rest of PR 6 validation work to the next container so the repo
   can be handed off in a clean state.

Files:

1. `src/IO/H5/Python/CompareFiniteRadiusRwz.py`
2. `support/Python/__main__.py`
3. `src/IO/H5/Python/CMakeLists.txt`
4. `tests/Unit/IO/H5/Python/Test_CompareFiniteRadiusRwz.py`
5. `tests/Unit/IO/H5/Python/CMakeLists.txt`

### Validation / Equivalence Notes

1. the RWZ implementation intentionally stays on the tensor-harmonic
   decomposition route used in the current SpECTRE extraction implementation,
   because a temporary scalar-harmonic rewrite was not yet validated to be
   convention-equivalent to SpEC,
2. the standard SpEC BBH RWZ path was traced and documented in this plan, and
   it uses a flat Minkowski background rather than a Schwarzschild background,
3. the current implementation is therefore still aimed at matching the
   standard SpEC BBH flat-background RWZ conventions first, before extending
   Newman-Penrose coverage.

## Deferred Housekeeping

1. run `black` on the Python files added or modified for RWZ work once a fresh
   container with a healthy Python toolchain is available,
2. after that, consider a broader repo-wide Python formatting pass if desired,
   but keep it separate from scientific validation commits.
