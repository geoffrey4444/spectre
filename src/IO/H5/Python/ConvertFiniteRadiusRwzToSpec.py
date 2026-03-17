#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

import os
import re
import shutil
from pathlib import Path

import click
import h5py
import numpy as np

import spectre.IO.H5 as spectre_h5
from spectre.IO.H5 import available_subfiles


_NATIVE_SUBFILE_PATTERN = re.compile(
    r"^(?P<target>[^/]+)/(?P<radius>[^/]+)/(?P<quantity>[^/]+)\.dat$"
)
_MODE_PATTERN = re.compile(r"Re\((?P<l>-?\d+),(?P<m>-?\d+)\)")
_OUTPUT_FILES = {
    "Strain": "rh_FiniteRadii_CodeUnits.h5",
    "PhiPlus": "PhiPlus_FiniteRadii_CodeUnits.h5",
    "PhiMinus": "PhiMinus_FiniteRadii_CodeUnits.h5",
}
_OUTPUT_DATASET_NAMES = {
    "Strain": "rh",
    "PhiPlus": "PhiPlus",
    "PhiMinus": "PhiMinus",
}
_METADATA_QUANTITIES = (
    "CoordRadius",
    "InitialAdmEnergy",
    "AverageLapse",
    "ArealRadius",
)


def _read_dat_file(open_h5_file: h5py.File, subfile: str):
    dataset = open_h5_file[subfile]
    return np.asarray(dataset), list(dataset.attrs["Legend"])


def _collect_native_rwz_data(h5_filename: str, target: str):
    with h5py.File(h5_filename, "r") as h5_file:
        native_subfiles = available_subfiles(h5_file, extension=".dat")
        grouped_subfiles = {}
        for subfile in native_subfiles:
            match = _NATIVE_SUBFILE_PATTERN.match(subfile)
            if match is None or match.group("target") != target:
                continue
            grouped_subfiles.setdefault(match.group("radius"), {})[
                match.group("quantity")
            ] = subfile

        if not grouped_subfiles:
            raise ValueError(
                f"Could not find native RWZ dat subfiles for target '{target}'"
                f" in '{h5_filename}'."
            )

        collected_data = {}
        for radius_name, subfiles in grouped_subfiles.items():
            missing_quantities = [
                quantity
                for quantity in (
                    *list(_OUTPUT_FILES.keys()),
                    *_METADATA_QUANTITIES,
                )
                if quantity not in subfiles
            ]
            if missing_quantities:
                raise ValueError(
                    f"Native RWZ radius group '{radius_name}' is missing"
                    f" required quantities: {missing_quantities}"
                )
            collected_data[radius_name] = {
                quantity: _read_dat_file(h5_file, subfile)
                for quantity, subfile in subfiles.items()
            }
        return collected_data


def _spec_radius_group(coord_radius: float) -> str:
    rounded_radius = int(round(coord_radius))
    if np.isclose(coord_radius, rounded_radius):
        return f"R{rounded_radius:04d}.dir"
    return f"R{coord_radius:.6f}".replace(".", "_") + ".dir"


def _mode_columns(legend):
    mode_columns = []
    for i in range(1, len(legend), 2):
        match = _MODE_PATTERN.fullmatch(legend[i])
        if match is None:
            raise ValueError(
                f"Could not parse native RWZ mode legend entry '{legend[i]}'."
            )
        mode_columns.append((int(match.group("l")), int(match.group("m")), i))
    return mode_columns


def _write_dat(output_filename: Path, subfile_name: str, legend, data):
    with spectre_h5.H5File(str(output_filename), "r+") as output_h5_file:
        dat_file = output_h5_file.insert_dat(
            path="/" + subfile_name.removesuffix(".dat"),
            legend=legend,
            version=0,
        )
        dat_file.append(data)


def convert_finite_radius_rwz_to_spec(
    h5_filename: str,
    output_dir: str = "GW2",
    target: str = "FiniteRadiusExtraction",
    force: bool = False,
):
    """Convert native finite-radius RWZ output to SpEC-compatible GW2 files.

    Reads the native finite-radius RWZ surface output written by
    'ObserveReggeWheelerZerilli' and writes exact SpEC-style finite-radius
    RWZ H5 files:

    - rh_FiniteRadii_CodeUnits.h5
    - PhiPlus_FiniteRadii_CodeUnits.h5
    - PhiMinus_FiniteRadii_CodeUnits.h5
    """

    native_data = _collect_native_rwz_data(h5_filename, target)
    output_dir_path = Path(output_dir)
    if output_dir_path.exists():
        if not force:
            raise ValueError(
                f"Output directory '{output_dir}' exists. Use '--force' to"
                " overwrite it."
            )
        shutil.rmtree(output_dir_path)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    output_files = {
        quantity: output_dir_path / filename
        for quantity, filename in _OUTPUT_FILES.items()
    }

    for radius_data in native_data.values():
        coord_radius_data, _ = radius_data["CoordRadius"]
        spec_radius_group = _spec_radius_group(coord_radius_data[0, 1])

        for metadata_quantity in _METADATA_QUANTITIES:
            metadata_data, metadata_legend = radius_data[metadata_quantity]
            if metadata_quantity == "InitialAdmEnergy":
                metadata_data = metadata_data[:1, :]
            for output_filename in output_files.values():
                _write_dat(
                    output_filename,
                    f"{spec_radius_group}/{metadata_quantity}.dat",
                    ["time", metadata_legend[1]],
                    metadata_data,
                )

        for quantity, output_filename in output_files.items():
            modal_data, modal_legend = radius_data[quantity]
            quantity_name = _OUTPUT_DATASET_NAMES[quantity]
            for l, m, column_index in _mode_columns(modal_legend):
                if l < 2:
                    continue
                mode_data = np.column_stack(
                    (
                        modal_data[:, 0],
                        modal_data[:, column_index],
                        modal_data[:, column_index + 1],
                    )
                )
                _write_dat(
                    output_filename,
                    f"{spec_radius_group}/Y_l{l}_m{m}.dat",
                    [
                        "time",
                        (
                            f"Re[{quantity_name}]_l{l}_m{m}"
                            f"(R={coord_radius_data[0, 1]:g})"
                        ),
                        (
                            f"Im[{quantity_name}]_l{l}_m{m}"
                            f"(R={coord_radius_data[0, 1]:g})"
                        ),
                    ],
                    mode_data,
                )

@click.command(
    name="convert-rwz-to-spec", help=convert_finite_radius_rwz_to_spec.__doc__
)
@click.argument(
    "h5_filename",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
)
@click.option(
    "--output-dir",
    "-o",
    default="GW2",
    show_default=True,
    type=click.Path(file_okay=False, dir_okay=True, writable=True),
    help="Directory where the SpEC-compatible GW2 files will be written.",
)
@click.option(
    "--target",
    "-t",
    default="FiniteRadiusExtraction",
    show_default=True,
    help="Native interpolation target group to convert.",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Overwrite the output directory if it already exists.",
)
def convert_finite_radius_rwz_to_spec_command(**kwargs):
    convert_finite_radius_rwz_to_spec(**kwargs)


if __name__ == "__main__":
    convert_finite_radius_rwz_to_spec_command(
        help_option_names=["-h", "--help"]
    )
