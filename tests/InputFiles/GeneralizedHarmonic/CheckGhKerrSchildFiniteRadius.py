#!/usr/bin/env python3
# Distributed under the MIT License.
# See LICENSE.txt for details.

import argparse
import os
import subprocess
from pathlib import Path

import h5py


def require_dataset(h5file, path):
    if path not in h5file:
        raise AssertionError(f"Missing dataset '{path}'")
    if not isinstance(h5file[path], h5py.Dataset):
        raise AssertionError(f"Expected dataset at '{path}'")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-filename", required=True)
    parser.add_argument("--run-directory", required=True)
    parser.add_argument("--cmake-source-directory", required=True)
    parser.add_argument("--cmake-bin-directory", required=True)
    args = parser.parse_args()

    run_directory = Path(args.run_directory)
    surface_file = run_directory / "GhKerrSchildSurfaces.h5"
    if not surface_file.exists():
        raise AssertionError(f"Missing surface file '{surface_file}'")

    with h5py.File(surface_file, "r") as h5file:
        require_dataset(
            h5file, "/FiniteRadiusExtraction/Radius2_1/CoordRadius.dat"
        )
        require_dataset(h5file, "/FiniteRadiusExtraction/Radius2_1/Strain.dat")
        require_dataset(h5file, "/FiniteRadiusExtraction/Radius2_1/PhiPlus.dat")
        require_dataset(
            h5file, "/FiniteRadiusExtraction/Radius2_1/PhiMinus.dat"
        )
        require_dataset(h5file, "/FiniteRadiusExtraction/Radius2_1/Psi4.dat")

    output_dir = run_directory / "GW2"
    subprocess.run(
        [
            str(Path(args.cmake_bin_directory) / "bin" / "python-spectre"),
            "-m",
            "spectre",
            "convert-rwz-to-spec",
            "--force",
            "--output-dir",
            str(output_dir),
            str(surface_file),
        ],
        check=True,
        env={**os.environ, "HDF5_USE_FILE_LOCKING": "FALSE"},
    )

    expected_files = [
        output_dir / "rh_FiniteRadii_CodeUnits.h5",
        output_dir / "PhiPlus_FiniteRadii_CodeUnits.h5",
        output_dir / "PhiMinus_FiniteRadii_CodeUnits.h5",
        output_dir / "rPsi4_FiniteRadii_CodeUnits.h5",
    ]
    for expected_file in expected_files:
        if not expected_file.exists():
            raise AssertionError(f"Missing converted file '{expected_file}'")


if __name__ == "__main__":
    main()
