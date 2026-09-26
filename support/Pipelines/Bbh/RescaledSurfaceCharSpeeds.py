# Distributed under the MIT License.
# See LICENSE.txt for details.

"""Evaluate rescaled-horizon characteristic speeds in saved volume data."""

import glob
from pathlib import Path

import numpy as np

import spectre.IO.H5 as spectre_h5
from spectre.ApparentHorizonFinder import (
    prepare_horizon_for_char_speeds,
    sample_rescaled_surface_char_speeds,
)
from spectre.DataStructures import DataVector
from spectre.DataStructures.Tensor import Frame, tnsr
from spectre.Domain import (
    deserialize_domain,
    deserialize_functions_of_time,
    strahlkorper_in_distorted_frame,
)
from spectre.IO.H5 import list_observations, open_volfiles
from spectre.IO.H5.IterElements import iter_elements
from spectre.IO.H5.TransformVolumeData import get_tensor_component_names
from spectre.Pipelines.Bbh.FindHorizon import find_horizon
from spectre.Strahlkorper import (
    Strahlkorper,
    read_surface_ylm_single_time,
    read_surface_ylm_single_time_distorted,
    time_deriv_of_strahlkorper,
)


def _matching_time(times, time, tolerance):
    matches = np.asarray(times)[
        np.abs(np.asarray(times) - time) <= tolerance * max(1.0, abs(time))
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one horizon at volume time {time:g}, found"
            f" {len(matches)} within tolerance {tolerance:g}."
        )
    return float(matches[0])


def _observation_geometry(h5_files, subfile_name, obs_id):
    for volume in open_volfiles(h5_files, subfile_name, obs_id):
        if volume.get_dimension() != 3:
            raise ValueError("The diagnostic requires three-dimensional data.")
        serialized_domain = volume.get_domain()
        if not serialized_domain:
            raise ValueError("The volume data must contain the saved domain.")
        domain = deserialize_domain[3](serialized_domain)
        functions_of_time = {}
        if domain.is_time_dependent():
            serialized_functions = volume.get_functions_of_time(obs_id)
            if not serialized_functions:
                raise ValueError(
                    "Moving domains require saved functions of time."
                )
            functions_of_time = deserialize_functions_of_time(
                serialized_functions
            )
        return domain, functions_of_time
    raise ValueError(f"No volume data found for observation {obs_id}.")


def _observation_fields(h5_files, subfile_name, obs_id, names, types):
    components = [
        component
        for name, tensor_type in zip(names, types)
        for component in get_tensor_component_names(name, tensor_type)
    ]

    def checked_volfiles():
        for volume in open_volfiles(h5_files, subfile_name, obs_id):
            missing = set(components) - set(
                volume.list_tensor_components(obs_id)
            )
            if missing:
                raise ValueError(
                    f"Missing volume tensor components: {sorted(missing)}"
                )
            yield volume

    element_ids, meshes = [], []
    tensors = [[] for _ in types]
    for element, data in iter_elements(
        checked_volfiles(),
        obs_id,
        tensor_components=components,
    ):
        element_ids.append(element.id)
        meshes.append(element.mesh)
        # Volume output may use Float precision, while DataVector tensors
        # require double buffers. Promote all fields, including Pi and Phi.
        data = np.asarray(data, dtype=np.float64)
        offset = 0
        for result, tensor_type in zip(tensors, types):
            result.append(tensor_type(data[offset : offset + tensor_type.size]))
            offset += tensor_type.size
    return element_ids, meshes, tensors


def rescaled_surface_char_speeds(
    h5_files,
    subfile_name,
    excision_sphere,
    *,
    horizons=None,
    horizon_file=None,
    horizon_subfile=None,
    horizon_frame=Frame.Inertial,
    initial_guess=None,
    fast_flow=None,
    observation_ids=None,
    number_of_surfaces=10,
    relative_excision_margin=1e-7,
    time_tolerance=1e-12,
    time_derivatives=None,
    spacetime_metric_name="SpacetimeMetric",
    horizon_tensor_names=None,
    horizon_data="Adm",
    pi_name="Pi",
    phi_name="Phi",
    blocks=None,
    output_file=None,
    output_subfile="RescaledCharSpeeds",
):
    """Sample the characteristic-speed diagnostic at volume observations.

    Supply exactly one source of horizons: a mapping ``horizons`` from times
    to Strahlkorpers, ``horizon_file`` with ``horizon_subfile`` containing saved
    Ylm coefficients, or an Inertial-frame ``initial_guess`` for re-finding
    horizons. Saved surfaces can be in the Inertial or Distorted frame. The
    surfaces are transformed at their observation times and recentered about
    the excision center before differentiating their coefficients.

    The derivative uses the current and up to two preceding saved horizons,
    or processed volume observations when re-finding horizons. Without earlier
    horizons the result has status MissingTimeDerivative unless
    ``time_derivatives`` supplies a derivative. This optional mapping contains
    Distorted-frame derivatives about the fixed excision center, indexed by
    volume time. Derivatives are never inferred to vanish from a single
    horizon. Sparse observation times can therefore differ from online
    diagnostics even with otherwise identical data.

    Arguments:
      h5_files: Volume H5 paths or a glob pattern. Inputs are read only.
      subfile_name: Volume-data subfile.
      excision_sphere: Name of the domain's excision sphere.
      horizons: Optional mapping of times to Inertial or Distorted surfaces.
      horizon_file: Optional H5 file containing horizon shape coefficients.
      horizon_subfile: Coefficient subfile, required with ``horizon_file``.
      horizon_frame: Frame of surfaces in ``horizon_file``.
      initial_guess: Optional Inertial surface for re-finding horizons. Each
        converged surface initializes the next observation's horizon find.
      fast_flow: Optional FastFlow solver, reset before each horizon find.
        Its tolerances should account for volume output precision.
      observation_ids: Optional observation IDs. They are processed in time
        order. Defaults to all observations.
      number_of_surfaces: Number of candidate surfaces, at least two.
      relative_excision_margin: Positive relative margin outside excision.
      time_tolerance: Horizon matching tolerance multiplied by max(1, |time|).
        Multiple matches are rejected; surfaces are not interpolated in time.
      time_derivatives: Optional mapping of volume times to supplied
        Distorted-frame derivative surfaces.
      spacetime_metric_name: Name of the inertial spacetime metric in volume
        data. This field, the domain, and functions of time suffice when
        horizons are supplied.
      horizon_tensor_names: Optional ADM tensor names passed to find_horizon.
      horizon_data: "Adm" uses saved inverse spatial metric, extrinsic
        curvature, and Christoffel symbols for horizon finding. "Gh" derives
        these fields at volume nodes from the spacetime metric, Pi, and Phi.
      pi_name: Name of Pi when horizon_data is "Gh".
      phi_name: Name of Phi when horizon_data is "Gh".
      blocks: Optional block names or groups covering the full candidate
        family. Defaults to blocks with the required Distorted-frame maps.
      output_file: Optional H5 file to append the results to.
      output_subfile: Dat subfile for the optional output.

    Returns: ``(legend, data)`` with a list of column names and a NumPy array
      with one row per observation. The columns, statuses, and speed
      convention are identical to the online diagnostic. Unavailable speeds
      are NaNs. Derived fields are computed at volume nodes before spatial
      interpolation, as in the online diagnostic.
    """
    if sum(x is not None for x in (horizons, horizon_file, initial_guess)) != 1:
        raise ValueError(
            "Specify exactly one of horizons, horizon_file, or initial_guess."
        )
    if (horizon_file is None) != (horizon_subfile is None):
        raise ValueError("Specify horizon_file and horizon_subfile together.")
    if horizon_frame not in (Frame.Inertial, Frame.Distorted):
        raise ValueError("Horizon frame must be Inertial or Distorted.")
    if not np.isfinite(time_tolerance) or time_tolerance < 0:
        raise ValueError("Time tolerance must be finite and nonnegative.")
    if number_of_surfaces < 2:
        raise ValueError("At least two surfaces are required.")
    if (
        not np.isfinite(relative_excision_margin)
        or relative_excision_margin <= 0
    ):
        raise ValueError("The relative excision margin must be positive.")
    if horizon_data not in ("Adm", "Gh"):
        raise ValueError("horizon_data must be 'Adm' or 'Gh'.")
    if initial_guess is not None and not isinstance(
        initial_guess, Strahlkorper[Frame.Inertial]
    ):
        raise ValueError("The initial guess must be in the Inertial frame.")
    if isinstance(h5_files, (str, Path)):
        h5_files = sorted(glob.glob(str(h5_files)))
    else:
        h5_files = list(h5_files)
    if not h5_files:
        raise ValueError("No volume files found.")
    obs_ids, obs_times = list_observations(
        open_volfiles(h5_files, subfile_name)
    )
    if observation_ids is not None:
        observation_ids = list(observation_ids)
        if len(set(observation_ids)) != len(observation_ids):
            raise ValueError("Observation IDs must not contain duplicates.")
        missing = set(observation_ids) - set(obs_ids)
        if missing:
            raise ValueError(f"Observation IDs not found: {sorted(missing)}")
    observations = sorted(
        (time, obs_id)
        for obs_id, time in zip(obs_ids, obs_times)
        if observation_ids is None or obs_id in observation_ids
    )
    if not observations:
        raise ValueError("No volume observations selected.")
    if len({time for time, _ in observations}) != len(observations):
        raise ValueError("Selected observations must have distinct times.")
    if horizons is not None:
        horizon_times = list(horizons)
    elif horizon_file is not None:
        with spectre_h5.H5File(str(horizon_file), "r") as h5file:
            data = np.asarray(h5file.get_dat(horizon_subfile).get_data())
            horizon_times = data[:, 0]
        read_surface = (
            read_surface_ylm_single_time
            if horizon_frame == Frame.Inertial
            else read_surface_ylm_single_time_distorted
        )
    metric_type = tnsr.aa[DataVector, 3, Frame.Inertial]
    if initial_guess is None:
        horizon_times = sorted(horizon_times)
        if not np.isfinite(horizon_times).all():
            raise ValueError("Horizon times must be finite.")
        if len(set(horizon_times)) != len(horizon_times):
            raise ValueError("Horizon times must be distinct.")
    history = []
    prepared_horizons = {}
    rows = []
    for time, obs_id in observations:
        domain, functions_of_time = _observation_geometry(
            h5_files, subfile_name, obs_id
        )
        field_names = [spacetime_metric_name]
        field_types = [metric_type]
        if initial_guess is not None and horizon_data == "Gh":
            field_names.extend([pi_name, phi_name])
            field_types.extend([metric_type, tnsr.iaa[DataVector, 3]])
        element_ids, meshes, fields = _observation_fields(
            h5_files, subfile_name, obs_id, field_names, field_types
        )
        if initial_guess is not None:
            if fast_flow is not None:
                fast_flow.reset_for_next_find()
            interpolation_options = {}
            if horizon_data == "Gh":
                from spectre.Pipelines.Bbh.HorizonInterpolation import (
                    GhHorizonInterpolator,
                )

                interpolation_options["interpolate_tensors"] = (
                    GhHorizonInterpolator(
                        domain,
                        time,
                        functions_of_time,
                        element_ids,
                        meshes,
                        *fields,
                    )
                )
            horizon, _ = find_horizon(
                h5_files,
                subfile_name,
                obs_id,
                time,
                initial_guess,
                fast_flow=fast_flow,
                tensor_names=horizon_tensor_names,
                compute_horizon_quantities=False,
                **interpolation_options,
            )
            initial_guess = horizon
            unprepared_history = [(time, horizon)]
        else:
            matched_time = _matching_time(horizon_times, time, time_tolerance)
            history_times = [
                horizon_time
                for horizon_time in horizon_times
                if horizon_time <= matched_time
            ][-3:]
            if time_derivatives is not None and time in time_derivatives:
                history_times = [matched_time]
            unprepared_history = [
                (
                    horizon_time,
                    (
                        horizons[horizon_time]
                        if horizons is not None
                        else read_surface(
                            str(horizon_file),
                            horizon_subfile,
                            horizon_time,
                            0.0,
                            True,
                        )
                    ),
                )
                for horizon_time in history_times
            ]
            history = []
        for horizon_time, horizon in unprepared_history:
            if horizon_time not in prepared_horizons:
                if isinstance(horizon, Strahlkorper[Frame.Inertial]):
                    try:
                        horizon = strahlkorper_in_distorted_frame(
                            horizon, domain, functions_of_time, horizon_time
                        )
                    except (RuntimeError, ValueError) as error:
                        raise ValueError(
                            "Cannot transform horizon at time"
                            f" {horizon_time:g}. Saved domain/maps must cover"
                            " this time; supply Distorted-frame history or an"
                            " explicit derivative if earlier maps are absent."
                        ) from error
                elif not isinstance(horizon, Strahlkorper[Frame.Distorted]):
                    raise ValueError(
                        "Horizons must be Inertial or Distorted surfaces."
                    )
                prepared_horizons[horizon_time] = (
                    prepare_horizon_for_char_speeds(
                        horizon, domain, excision_sphere
                    )
                )
            history.insert(0, (horizon_time, prepared_horizons[horizon_time]))
        del history[3:]
        horizon = history[0][1]
        if time_derivatives is not None and time in time_derivatives:
            derivative = time_derivatives[time]
        elif len(history) > 1:
            derivative = time_deriv_of_strahlkorper(history)
        else:
            derivative = None
        legend, row = sample_rescaled_surface_char_speeds(
            horizon,
            derivative,
            element_ids,
            meshes,
            fields[0],
            domain,
            functions_of_time,
            time,
            excision_sphere,
            number_of_surfaces=number_of_surfaces,
            relative_excision_margin=relative_excision_margin,
            blocks=blocks,
        )
        rows.append(row)
    data = np.asarray(rows)
    if output_file is not None:
        with spectre_h5.H5File(str(output_file), "a") as h5file:
            output = h5file.try_insert_dat(output_subfile, legend, 0)
            if output.get_legend() != legend:
                raise ValueError("Existing output has a different legend.")
            output.append(data)
    return legend, data
