#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

import click
import h5py
import numpy as np

from spectre.IO.H5 import available_subfiles


def compare_finite_radius_rwz(
    reference_h5: str,
    candidate_h5: str,
    absolute_tolerance: float = 0.0,
    relative_tolerance: float = 0.0,
):
    """Compare two SpEC-style finite-radius RWZ H5 files mode-by-mode."""

    with h5py.File(reference_h5, "r") as reference_file, h5py.File(
        candidate_h5, "r"
    ) as candidate_file:
        reference_subfiles = available_subfiles(
            reference_file, extension=".dat"
        )
        candidate_subfiles = available_subfiles(
            candidate_file, extension=".dat"
        )
        if reference_subfiles != candidate_subfiles:
            raise ValueError(
                "RWZ files do not contain the same dat subfiles.\n"
                f"Reference: {reference_subfiles}\n"
                f"Candidate: {candidate_subfiles}"
            )

        for subfile in reference_subfiles:
            reference_dataset = reference_file[subfile]
            candidate_dataset = candidate_file[subfile]
            reference_legend = list(reference_dataset.attrs["Legend"])
            candidate_legend = list(candidate_dataset.attrs["Legend"])
            if reference_legend != candidate_legend:
                raise ValueError(
                    f"Legend mismatch for '{subfile}'.\n"
                    f"Reference: {reference_legend}\n"
                    f"Candidate: {candidate_legend}"
                )
            np.testing.assert_allclose(
                np.asarray(candidate_dataset),
                np.asarray(reference_dataset),
                atol=absolute_tolerance,
                rtol=relative_tolerance,
                err_msg=f"Mismatch in subfile '{subfile}'",
            )


@click.command(name="compare-rwz", help=compare_finite_radius_rwz.__doc__)
@click.argument(
    "reference_h5",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
)
@click.argument(
    "candidate_h5",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
)
@click.option(
    "--absolute-tolerance",
    default=0.0,
    show_default=True,
    type=float,
    help="Absolute tolerance for mode-by-mode data comparison.",
)
@click.option(
    "--relative-tolerance",
    default=0.0,
    show_default=True,
    type=float,
    help="Relative tolerance for mode-by-mode data comparison.",
)
def compare_finite_radius_rwz_command(**kwargs):
    compare_finite_radius_rwz(**kwargs)


if __name__ == "__main__":
    compare_finite_radius_rwz_command(help_option_names=["-h", "--help"])
