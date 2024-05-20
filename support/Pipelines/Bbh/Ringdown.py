# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
import warnings
from pathlib import Path
from typing import Optional, Union

import click
import numpy as np
import yaml
from rich.pretty import pretty_repr

import spectre.Evolution.Ringdown as Ringdown
import spectre.IO.H5 as spectre_h5
from spectre.DataStructures import DataVector
from spectre.Domain import deserialize_functions_of_time
from spectre.support.Schedule import schedule, scheduler_options

logger = logging.getLogger(__name__)

RINGDOWN_INPUT_FILE_TEMPLATE = Path(__file__).parent / "Ringdown.yaml"


def cubic(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d


def dt_cubic(x, a, b, c, d):
    return 3 * a * x**2 + 2 * b * x + c


def dt2_cubic(x, a, b, c, d):
    return 6 * a * x + 2 * b


# Cubic fit transformed coefs to get first and second time derivatives
def fit_to_a_cubic(times, coefs, match_time, zero_coefs):
    fits = []
    fit_ahc = []
    fit_dt_ahc = []
    fit_dt2_ahc = []
    for j in np.arange(0, coefs.shape[-1], 1):
        # Optionally, avoid fitting coefficients sufficiently close to zero by
        # just setting these coefficients and their time derivatives to zero.
        if zero_coefs is not None and sum(np.abs(coefs[:, j])) < zero_coefs:
            fits.append(np.zeros(4))
            fit_ahc.append(0.0)
            fit_dt_ahc.append(0.0)
            fit_dt2_ahc.append(0.0)
            continue
        # Ignore RankWarnings suggesting the fit might not be good enough;
        # for equal-mass non-spinning, sufficiently good fits for starting
        # a ringdown, even though RankWarnings were triggered
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", np.RankWarning)
            fit = np.polyfit(times, coefs[:, j], 3)
        fits.append(fit)
        fit_ahc.append(cubic(match_time, *(fit)))
        fit_dt_ahc.append(dt_cubic(match_time, *(fit)))
        fit_dt2_ahc.append(dt2_cubic(match_time, *(fit)))

    return fit_ahc, fit_dt_ahc, fit_dt2_ahc


def compute_ahc_coefs_in_ringdown_distorted_frame(
    path_to_ah_h5,
    ahc_subfile_path,
    path_to_fot_h5,
    fot_subfile_path,
    path_to_output_h5,
    output_subfile_prefix,
    number_of_steps,
    which_obs_id,
    settling_timescale,
    zero_coefs,
):
    shape_coefs = {}

    output_subfile_ahc = output_subfile_prefix + "AhC_Ylm"
    output_subfile_dt_ahc = output_subfile_prefix + "dtAhC_Ylm"
    output_subfile_dt2_ahc = output_subfile_prefix + "dt2AhC_Ylm"
    if ahc_subfile_path.split(".")[-1] == "dat":
        ahc_subfile_path = ahc_subfile_path.split(".")[0]

    ahc_times = []
    ahc_legend = ""
    ahc_center = []
    ahc_lmax = 0
    with spectre_h5.H5File(path_to_ah_h5, "r") as h5file:
        datfile = h5file.get_dat("/" + ahc_subfile_path)
        datfile_np = np.array(datfile.get_data())
        ahc_times = datfile_np[:, 0]
        ahc_legend = datfile.get_legend()
        ahc_center = [datfile_np[0][1], datfile_np[0][2], datfile_np[0][3]]
        ahc_lmax = int(datfile_np[0][4])

    # dict storing other info for starting a ringdown obtained in the process
    # of getting the ringdown-distorted-frame AhC coefficients
    info_for_ringdown = {}
    legend_for_ringdown = {}

    # Transform AhC coefs to ringdown distorted frame and get other data
    # needed to start a ringdown, such as initial values for functions of time
    with spectre_h5.H5File(path_to_fot_h5, "r") as h5file:
        if fot_subfile_path.split(".")[-1] == "vol":
            fot_subfile_path = fot_subfile_path.split(".")[0]
        volfile = h5file.get_vol("/" + fot_subfile_path)
        obs_ids = volfile.list_observation_ids()
        fot_times = list(map(volfile.get_observation_value, obs_ids))
        serialized_fots = volfile.get_functions_of_time(obs_ids[which_obs_id])
        functions_of_time = deserialize_functions_of_time(serialized_fots)

        # The inspiral expansion map includes two functions of time:
        # an expansion map allowing the black holes to move closer together in
        # comoving coordinates, and an expansion map causing the outer boundary
        # to move slightly inwards. The ringdown only requires the outer
        # boundary expansion map, so set the other map to the identity.
        exp_func_and_2_derivs = [1.0, 0.0, 0.0]

        exp_outer_bdry_func_and_2_derivs = [
            x[0]
            for x in functions_of_time[
                "ExpansionOuterBoundary"
            ].func_and_2_derivs(fot_times[which_obs_id])
        ]
        rot_func_and_2_derivs_tuple = functions_of_time[
            "Rotation"
        ].func_and_2_derivs(fot_times[which_obs_id])
        rot_func_and_2_derivs = [
            [coef for coef in x] for x in rot_func_and_2_derivs_tuple
        ]

        match_time = fot_times[which_obs_id]

        coefs_at_different_times = np.array(
            Ringdown.strahlkorper_coefs_in_ringdown_distorted_frame(
                path_to_ah_h5,
                ahc_subfile_path,
                ahc_times,
                number_of_steps,
                match_time,
                settling_timescale,
                exp_func_and_2_derivs,
                exp_outer_bdry_func_and_2_derivs,
                rot_func_and_2_derivs,
            )
        )

        # Print out coefficients for insertion into BBH domain
        logger.debug("Expansion: ", exp_func_and_2_derivs)
        logger.debug("ExpansionOutrBdry: ", exp_outer_bdry_func_and_2_derivs)
        logger.debug("Rotation: ", rot_func_and_2_derivs)
        logger.debug("Match time: ", match_time)
        logger.debug("Settling timescale: ", settling_timescale)
        logger.debug("Lmax: ", ahc_lmax)

        shape_coefs["Expansion"]: exp_func_and_2_derivs
        shape_coefs["ExpansionOutrBdry"]: exp_outer_bdry_func_and_2_derivs
        shape_coefs["Rotation"]: rot_func_and_2_derivs
        shape_coefs["Match time"]: match_time
        shape_coefs["Settling timescale"]: settling_timescale
        shape_coefs["Lmax"]: ahc_lmax

        legend_fot = ["FunctionOfTime", "dtFunctionOfTime", "dt2FunctionOfTime"]
        legend_value = ["Value"]
        info_for_ringdown["Expansion"] = exp_func_and_2_derivs
        legend_for_ringdown["Expansion"] = legend_fot
        info_for_ringdown["ExpansionOuterBdry"] = (
            exp_outer_bdry_func_and_2_derivs
        )
        legend_for_ringdown["ExpansionOuterBdry"] = legend_fot
        info_for_ringdown["Rotation0"] = [
            rot_func_and_2_derivs[0][0],
            rot_func_and_2_derivs[1][0],
            rot_func_and_2_derivs[2][0],
        ]
        info_for_ringdown["Rotation1"] = [
            rot_func_and_2_derivs[0][1],
            rot_func_and_2_derivs[1][1],
            rot_func_and_2_derivs[2][1],
        ]
        info_for_ringdown["Rotation2"] = [
            rot_func_and_2_derivs[0][2],
            rot_func_and_2_derivs[1][2],
            rot_func_and_2_derivs[2][2],
        ]
        info_for_ringdown["Rotation3"] = [
            rot_func_and_2_derivs[0][3],
            rot_func_and_2_derivs[1][3],
            rot_func_and_2_derivs[2][3],
        ]
        legend_for_ringdown["Rotation0"] = legend_fot
        legend_for_ringdown["Rotation1"] = legend_fot
        legend_for_ringdown["Rotation2"] = legend_fot
        legend_for_ringdown["Rotation3"] = legend_fot
        info_for_ringdown["MatchTime"] = [match_time]
        legend_for_ringdown["MatchTime"] = legend_value
        info_for_ringdown["SettlingTimescale"] = [settling_timescale]
        legend_for_ringdown["SettlingTimescale"] = legend_value
        info_for_ringdown["Lmax"] = [ahc_lmax]
        legend_for_ringdown["Lmax"] = legend_value

    # Do not include AhCs at times greater than the match time. Errors tend
    # to grow as time increases, so fit derivatives using the match time
    # and earlier times, to get a more accurate fit.
    ahc_times_for_fit_list = []
    coefs_at_different_times_for_fit_list = []
    for i, time in enumerate(ahc_times[-number_of_steps:]):
        if time <= match_time:
            ahc_times_for_fit_list.append(time)
            coefs_at_different_times_for_fit_list.append(
                coefs_at_different_times[i]
            )
    ahc_times_for_fit = np.array(ahc_times_for_fit_list)
    coefs_at_different_times_for_fit = np.array(
        coefs_at_different_times_for_fit_list
    )
    if ahc_times_for_fit.shape[0] == 0:
        logger.warning(
            "No available AhC times before selected match time; using all"
            " available AhC times, even though numerical errors are likely"
            " larger after the match time"
        )
        ahc_times_for_fit = ahc_times[-number_of_steps:]
        coefs_at_different_times_for_fit = coefs_at_different_times[
            -number_of_steps:
        ]

    logger.debug("AhC times available: " + str(ahc_times.shape[0]))
    logger.debug(
        "AhC available time range: "
        + str(np.min(ahc_times))
        + " - "
        + str(np.max(ahc_times))
    )
    logger.debug("AhC times used: " + str(ahc_times_for_fit.shape[0]))
    logger.debug(
        "AhC used time range: "
        + str(np.min(ahc_times_for_fit))
        + " - "
        + str(np.max(ahc_times_for_fit))
    )
    logger.debug(
        "Coef times available: " + str(coefs_at_different_times.shape[0])
    )
    logger.debug(
        "Coef times used: " + str(coefs_at_different_times_for_fit.shape[0])
    )

    fit_ahc_coefs, fit_ahc_dt_coefs, fit_ahc_dt2_coefs = fit_to_a_cubic(
        ahc_times_for_fit,
        coefs_at_different_times_for_fit,
        match_time,
        zero_coefs,
    )

    # output coefs to H5
    # Note: assumes no translation, so inertial and distorted centers are the
    # same, i.e. are both the origin. A future update will incorporate
    # translation.
    ahc_legend[1] = ahc_legend[1].replace("Inertial", "Distorted")
    ahc_legend[2] = ahc_legend[2].replace("Inertial", "Distorted")
    ahc_legend[3] = ahc_legend[3].replace("Inertial", "Distorted")

    fit_ahc_coefs_dv = -DataVector(fit_ahc_coefs)
    fit_ahc_dt_coefs_dv = -DataVector(fit_ahc_dt_coefs)
    fit_ahc_dt2_coefs_dv = -DataVector(fit_ahc_dt2_coefs)
    fit_ahc_coefs_to_write = Ringdown.wrap_fill_ylm_data(
        fit_ahc_coefs_dv, match_time, ahc_center, ahc_lmax
    )
    fit_ahc_dt_coefs_to_write = Ringdown.wrap_fill_ylm_data(
        fit_ahc_dt_coefs_dv, match_time, ahc_center, ahc_lmax
    )
    fit_ahc_dt2_coefs_to_write = Ringdown.wrap_fill_ylm_data(
        fit_ahc_dt2_coefs_dv, match_time, ahc_center, ahc_lmax
    )
    with spectre_h5.H5File(file_name=path_to_output_h5, mode="a") as h5file:
        ahc_datfile = h5file.insert_dat(
            path="/" + output_subfile_ahc, legend=ahc_legend, version=0
        )
        ahc_datfile.append(fit_ahc_coefs_to_write)

    with spectre_h5.H5File(file_name=path_to_output_h5, mode="a") as h5file:
        ahc_dt_datfile = h5file.insert_dat(
            path="/" + output_subfile_dt_ahc, legend=ahc_legend, version=0
        )
        ahc_dt_datfile.append(fit_ahc_dt_coefs_to_write)

    with spectre_h5.H5File(file_name=path_to_output_h5, mode="a") as h5file:
        ahc_dt2_datfile = h5file.insert_dat(
            path="/" + output_subfile_dt2_ahc, legend=ahc_legend, version=0
        )
        ahc_dt2_datfile.append(fit_ahc_dt2_coefs_to_write)

    return shape_coefs, info_for_ringdown


def ringdown_parameters(
    inspiral_input_file: dict,
    inspiral_run_dir: Union[str, Path],
    refinement_level: int,
    polynomial_order: int,
) -> dict:
    """Determine ringdown parameters from the inspiral.

    These parameters fill the 'RINGDOWN_INPUT_FILE_TEMPLATE'.

    Arguments:
      inspiral_input_file: Inspiral input file as a dictionary.
      id_run_dir: Directory of the inspiral run. Paths in the input file
        are relative to this directory.
      refinement_level: h-refinement level.
      polynomial_order: p-refinement level.
    """
    return {
        # Initial data files
        "IdFileGlob": str(
            Path(inspiral_run_dir).resolve()
            / (inspiral_input_file["Observers"]["VolumeFileName"] + "*.h5")
        ),
        # Resolution
        "L": refinement_level,
        "P": polynomial_order,
    }


def start_ringdown(
    path_to_ah_h5: Union[str, Path],
    ahc_subfile_path: str,
    path_to_fot_h5: Union[str, Path],
    fot_subfile_path: str,
    path_to_output_h5: Union[str, Path],
    output_subfile_prefix: str,
    number_of_steps: int,
    match_time: float,
    settling_timescale: float,
    zero_coefs: float,
    inspiral_input_file_path: Union[str, Path],
    refinement_level: int,
    polynomial_order: int,
    inspiral_run_dir: Optional[Union[str, Path]] = None,
    ringdown_input_file_template: Union[
        str, Path
    ] = RINGDOWN_INPUT_FILE_TEMPLATE,
    pipeline_dir: Optional[Union[str, Path]] = None,
    run_dir: Optional[Union[str, Path]] = None,
    segments_dir: Optional[Union[str, Path]] = None,
    **scheduler_kwargs,
):
    """Schedule a ringdown simulation from the inspiral.

    Point the INSPIRAL_INPUT_FILE_PATH to the input file of the last inspiral
    segment. Also specify 'inspiral_run_dir' if the simulation was run in a
    different directory than where the input file is. Parameters for the
    ringdown will be determined from the inspiral and inserted into the
    'ringdown_input_file_template'. The remaining options are forwarded to the
    'schedule' command. See 'schedule' docs for details.

    Here 'parameters for the ringdown' includes the information needed to
    initialize the time-dependent maps, including the shape me. Common horizon
    shape coefficients in the ringdown distorted frame will be written to
    disk that the ringdown input file will point to.
    """
    logger.warning(
        "The BBH pipeline is still experimental. Please review the"
        " generated input files. In particular, the ringdown BBH pipline has"
        " been tested for a q=1, spin=0 quasicircular inspiral but does not"
        " yet support accounting for a nonzero translation map in the inspiral"
        " (necessary for unequal-mass mergers.)"
    )

    # Determine ringdown parameters from inspiral
    with open(inspiral_input_file_path, "r") as open_input_file:
        _, inspiral_input_file = yaml.safe_load_all(open_input_file)
    if inspiral_run_dir is None:
        inspiral_run_dir = Path(inspiral_input_file_path).resolve().parent

    # Resolve directories
    if pipeline_dir:
        pipeline_dir = Path(pipeline_dir).resolve()
    if pipeline_dir and not segments_dir and not run_dir:
        segments_dir = pipeline_dir / "003_Ringdown"

    ringdown_params = ringdown_parameters(
        inspiral_input_file,
        inspiral_run_dir,
        refinement_level=refinement_level,
        polynomial_order=polynomial_order,
    )
    logger.debug(f"Ringdown parameters: {pretty_repr(ringdown_params)}")

    # Compute ringdown shape coefficients and function of time info
    # for ringdown
    with spectre_h5.H5File(str(path_to_fot_h5), "r") as h5file:
        if fot_subfile_path.split(".")[-1] == "vol":
            fot_subfile_path = fot_subfile_path.split(".")[0]
        volfile = h5file.get_vol("/" + fot_subfile_path)
        obs_ids = volfile.list_observation_ids()
        fot_times = np.array(list(map(volfile.get_observation_value, obs_ids)))
        which_obs_id = np.argmin(np.abs(fot_times - match_time))

        logger.debug("Desired match time: " + str(match_time))
        logger.debug("Selected ObservationID: " + str(which_obs_id))
        logger.debug("Selected match time: " + str(fot_times[which_obs_id]))
        serialized_fots = volfile.get_functions_of_time(obs_ids[which_obs_id])
        functions_of_time = deserialize_functions_of_time(serialized_fots)
        logger.debug("Succeeded in reading in functions of time:")
        logger.debug(functions_of_time)

    coefs, fot_info = compute_ahc_coefs_in_ringdown_distorted_frame(
        str(inspiral_run_dir) + "/" + str(path_to_ah_h5),
        ahc_subfile_path,
        str(inspiral_run_dir) + "/" + str(path_to_fot_h5),
        fot_subfile_path,
        str(inspiral_run_dir) + "/" + str(path_to_output_h5),
        output_subfile_prefix,
        number_of_steps,
        which_obs_id,
        settling_timescale,
        zero_coefs,
    )

    # Add additional parameters to substitute in ringdown template
    # Primarily, these will initialize functions of time
    extra_params = {}

    # Schedule!
    return schedule(
        ringdown_input_file_template,
        **ringdown_params,
        **scheduler_kwargs,
        pipeline_dir=pipeline_dir,
        run_dir=run_dir,
        segments_dir=segments_dir,
    )


@click.command(name="start-ringdown", help=start_ringdown.__doc__)
@click.option(
    "--path_to_ah_h5",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        path_type=Path,
    ),
    help="Path of file containing AhC coefs",
)
@click.option(
    "--ahc_subfile_path",
    type=str,
    required=True,
    help="Subfile path to reduction data containing AhC coefs",
)
@click.option(
    "--path_to_fot_h5",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        path_type=Path,
    ),
    help="Path to file containing funcs of time",
)
@click.option(
    "--fot_subfile_path",
    type=str,
    required=True,
    help="Subfile path to volume data with funcs of time at different times",
)
@click.option(
    "--path_to_output_h5",
    type=click.Path(
        exists=False,
        file_okay=True,
        dir_okay=False,
        readable=True,
        path_type=Path,
    ),
    help="Path relative to inspiral-run-dir to shape coefs output h5 file",
)
@click.option(
    "--output_subfile_prefix",
    type=str,
    required=True,
    help="Output subfile prefix for AhC coefs",
)
@click.option(
    "--number_of_steps",
    type=int,
    required=True,
    help="Number of steps from end to look for AhC data",
)
@click.option(
    "--match_time",
    required=True,
    type=float,
    help="Desired match time (volume data must contain data at this time)",
)
@click.option(
    "--settling_timescale",
    required=True,
    type=float,
    help="Damping timescale for settle to const",
)
@click.option(
    "--zero-coefs",
    type=float,
    default=None,
    help="What value of coefs to zero below. None means don't zero any",
)
@click.argument(
    "inspiral_input_file_path",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        path_type=Path,
    ),
)
@click.option(
    "-i",
    "--inspiral-run-dir",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        path_type=Path,
    ),
    help=(
        "Directory of the last inspiral segment. Paths in the input file are"
        " relative to this directory."
    ),
    show_default="directory of the INSPIRAL_INPUT_FILE_PATH",
)
@click.option(
    "--ringdown-input-file-template",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        path_type=Path,
    ),
    default=RINGDOWN_INPUT_FILE_TEMPLATE,
    help="Input file template for the ringdown.",
    show_default=True,
)
@click.option(
    "--refinement-level",
    "-L",
    type=int,
    help="h-refinement level.",
    default=2,
    show_default=True,
)
@click.option(
    "--polynomial-order",
    "-P",
    type=int,
    help="p-refinement level.",
    default=11,
    show_default=True,
)
@click.option(
    "--pipeline-dir",
    "-d",
    type=click.Path(
        writable=True,
        path_type=Path,
    ),
    help="Directory where steps in the pipeline are created.",
)
@scheduler_options
def start_ringdown_command(**kwargs):
    _rich_traceback_guard = True  # Hide traceback until here
    start_ringdown(**kwargs)


if __name__ == "__main__":
    start_ringdown_command(help_option_names=["-h", "--help"])
