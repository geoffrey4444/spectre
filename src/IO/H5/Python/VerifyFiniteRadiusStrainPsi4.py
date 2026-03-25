#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

import re

import click
import h5py
import numpy as np

_RADIUS_GROUP_PATTERN = re.compile(r"^R(?P<radius>[0-9_]+)\.dir$")
_MODE_DATASET_PATTERN = re.compile(r"^Y_l(?P<l>-?\d+)_m(?P<m>-?\d+)\.dat$")


def _parse_radius(group_name: str) -> float:
    match = _RADIUS_GROUP_PATTERN.fullmatch(group_name)
    if match is None:
        raise ValueError(f"Could not parse finite-radius group '{group_name}'.")
    return float(match.group("radius").replace("_", "."))


def _outermost_common_radius_group(
    strain_file: h5py.File, psi4_file: h5py.File
) -> str:
    strain_groups = {
        key
        for key in strain_file.keys()
        if _RADIUS_GROUP_PATTERN.fullmatch(key)
    }
    psi4_groups = {
        key for key in psi4_file.keys() if _RADIUS_GROUP_PATTERN.fullmatch(key)
    }
    common_groups = strain_groups & psi4_groups
    if not common_groups:
        raise ValueError(
            "No common finite-radius groups were found between the strain "
            "and Psi4 files."
        )
    return max(common_groups, key=_parse_radius)


def _mode_datasets(radius_group: h5py.Group):
    return {
        key
        for key in radius_group.keys()
        if _MODE_DATASET_PATTERN.fullmatch(key) is not None
    }


def _second_derivative(values: np.ndarray, time: np.ndarray) -> np.ndarray:
    dt_plus = time[2:] - time[1:-1]
    dt_minus = time[1:-1] - time[:-2]
    return 2.0 * (
        values[:-2] / (dt_minus * (dt_plus + dt_minus))
        - values[1:-1] / (dt_plus * dt_minus)
        + values[2:] / (dt_plus * (dt_plus + dt_minus))
    )


def verify_finite_radius_strain_psi4(
    strain_h5: str,
    psi4_h5: str,
    absolute_tolerance: float = 0.0,
    relative_tolerance: float = 0.0,
):
    """Verify the finite-radius relation Psi4 ~= -ddot(h) on the outermost radius.

    Both inputs must use the standard SpEC-style finite-radius waveform layout.
    The check is performed mode-by-mode on the outermost extraction radius
    present in both files.
    """

    with h5py.File(strain_h5, "r") as strain_file, h5py.File(
        psi4_h5, "r"
    ) as psi4_file:
        radius_group_name = _outermost_common_radius_group(
            strain_file, psi4_file
        )
        strain_group = strain_file[radius_group_name]
        psi4_group = psi4_file[radius_group_name]

        common_mode_datasets = _mode_datasets(strain_group) & _mode_datasets(
            psi4_group
        )
        if not common_mode_datasets:
            raise ValueError(
                "No common waveform mode datasets were found in the outermost "
                f"radius group '{radius_group_name}'."
            )

        for dataset_name in sorted(common_mode_datasets):
            strain_dataset = np.asarray(strain_group[dataset_name])
            psi4_dataset = np.asarray(psi4_group[dataset_name])
            if strain_dataset.shape[0] < 3 or psi4_dataset.shape[0] < 3:
                raise ValueError(
                    f"Dataset '{radius_group_name}/{dataset_name}' must have "
                    "at least three time samples."
                )
            if strain_dataset.shape[1] != 3 or psi4_dataset.shape[1] != 3:
                raise ValueError(
                    f"Dataset '{radius_group_name}/{dataset_name}' must have "
                    "three columns: time, real part, imaginary part."
                )
            ddot_real = _second_derivative(
                strain_dataset[:, 1], strain_dataset[:, 0]
            )
            ddot_imag = _second_derivative(
                strain_dataset[:, 2], strain_dataset[:, 0]
            )
            np.testing.assert_allclose(
                strain_dataset[1:-1, 0],
                psi4_dataset[:, 0],
                atol=absolute_tolerance,
                rtol=relative_tolerance,
                err_msg=(
                    "Time columns differ between interior strain samples and "
                    f"Psi4 for '{radius_group_name}/{dataset_name}'."
                ),
            )
            predicted_psi4 = np.column_stack((-ddot_real, -ddot_imag))
            np.testing.assert_allclose(
                psi4_dataset[:, 1:3],
                predicted_psi4,
                atol=absolute_tolerance,
                rtol=relative_tolerance,
                err_msg=(
                    "Finite-radius Psi4 does not match -ddot(h) for "
                    f"'{radius_group_name}/{dataset_name}'."
                ),
            )


@click.command(
    name="verify-h-psi4", help=verify_finite_radius_strain_psi4.__doc__
)
@click.argument(
    "strain_h5",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
)
@click.argument(
    "psi4_h5",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
)
@click.option(
    "--absolute-tolerance",
    default=0.0,
    show_default=True,
    type=float,
    help="Absolute tolerance for the Psi4 ~= -ddot(h) comparison.",
)
@click.option(
    "--relative-tolerance",
    default=0.0,
    show_default=True,
    type=float,
    help="Relative tolerance for the Psi4 ~= -ddot(h) comparison.",
)
def verify_finite_radius_strain_psi4_command(**kwargs):
    verify_finite_radius_strain_psi4(**kwargs)


if __name__ == "__main__":
    verify_finite_radius_strain_psi4_command(help_option_names=["-h", "--help"])
