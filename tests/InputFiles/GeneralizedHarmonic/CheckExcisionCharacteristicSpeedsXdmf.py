#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

import argparse
import os
import subprocess
import unittest
import xml.etree.ElementTree as ET

import h5py
import numpy as np


class CheckExcisionCharacteristicSpeedsXdmf(unittest.TestCase):
    def test_characteristic_speeds_and_coordinates(self):
        surface_file = os.path.join(
            self.run_directory, "GhKerrSchildSurfaces.h5"
        )
        subfile_name = "ExcisionBoundary"
        speed_names = [
            "CharacteristicSpeedMetric",
            "CharacteristicSpeedZero",
            "CharacteristicSpeedPlus",
            "CharacteristicSpeedMinus",
        ]
        with h5py.File(surface_file, "r") as open_surface_file:
            observations = open_surface_file[subfile_name + ".vol"]
            observation_groups = [
                item
                for name, item in observations.items()
                if name.startswith("ObservationId")
            ]
            self.assertGreater(len(observation_groups), 0)
            observation = min(
                observation_groups,
                key=lambda obs: obs.attrs["observation_value"],
            )
            for name in speed_names:
                self.assertIn(name, observation)
            for component in ["x", "y", "z"]:
                self.assertIn("InertialCoordinates_" + component, observation)

            # At the initial time the surface is r=1.9 in a stationary,
            # unit-mass Schwarzschild solution with gamma1=-1. The surface
            # normal points away from the hole, so all physical speeds are
            # negative inside the horizon. The metric interpolation uses only
            # five grid points per dimension, hence the modest tolerance.
            self.assertEqual(observation.attrs["observation_value"], 0.0)
            radius = 1.9
            lapse = np.sqrt(radius / (radius + 2.0))
            normal_dot_shift = 2.0 / np.sqrt(radius * (radius + 2.0))
            expected_speeds = [
                0.0,
                -normal_dot_shift,
                -normal_dot_shift + lapse,
                -normal_dot_shift - lapse,
            ]
            for name, expected in zip(speed_names, expected_speeds):
                np.testing.assert_allclose(
                    observation[name][:],
                    expected,
                    atol=1.0e-2,
                    rtol=1.0e-2,
                )

        xmf_output = os.path.join(self.run_directory, "ExcisionBoundary")
        subprocess.run(
            [
                os.path.join(self.cmake_bin_directory, "bin", "spectre"),
                "generate-xdmf",
                "--output",
                xmf_output,
                "--subfile-name",
                subfile_name,
                surface_file,
            ],
            check=True,
        )
        xmf_root = ET.parse(xmf_output + ".xmf").getroot()
        attributes = {
            attribute.get("Name"): attribute.get("AttributeType")
            for attribute in xmf_root.findall(".//Attribute")
        }
        for name in speed_names:
            self.assertEqual(attributes[name], "Scalar")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-filename")
    parser.add_argument("--run-directory")
    parser.add_argument("--cmake-source-directory")
    parser.add_argument("--cmake-bin-directory")
    duplicate_test_case, remaining_args = parser.parse_known_args(
        namespace=CheckExcisionCharacteristicSpeedsXdmf
    )
    del duplicate_test_case
    unittest.main(argv=[parser.prog] + remaining_args, verbosity=2)
