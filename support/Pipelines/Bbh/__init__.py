# Distributed under the MIT License.
# See LICENSE.txt for details.

import click

from .ComputeRingdownShapeCoefs import compute_ringdown_shape_coefs_command
from .InitialData import generate_id_command
from .Inspiral import start_inspiral_command
from .Ringdown import start_ringdown_command


@click.group(name="bbh")
def bbh_pipeline():
    """Pipeline for binary black hole simulations."""
    pass


bbh_pipeline.add_command(generate_id_command)
bbh_pipeline.add_command(start_inspiral_command)
bbh_pipeline.add_command(start_ringdown_command)
bbh_pipeline.add_command(compute_ringdown_shape_coefs_command)

if __name__ == "__main__":
    bbh_pipeline(help_option_names=["-h", "--help"])
