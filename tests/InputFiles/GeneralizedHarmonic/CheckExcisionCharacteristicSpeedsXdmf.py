#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

import argparse
import os
import subprocess
import unittest
import xml.etree.ElementTree as ET

import h5py


class CheckExcisionCharacteristicSpeedsXdmf(unittest.TestCase):
    def test_characteristic_speeds_and_coordinates(self):
        surface_file = os.path.join(
            self.run_directory, "GhKerrSchildSurfaces.h5"
        )
        subfile_name = "ExcisionBoundary"
        with h5py.File(surface_file, "r") as open_surface_file:
            observations = open_surface_file[subfile_name + ".vol"]
            observation = observations[next(iter(observations))]
            for component in ["t", "x", "y", "z"]:
                self.assertIn(
                    "CharacteristicSpeedsOnStrahlkorper_" + component,
                    observation,
                )
            for component in ["x", "y", "z"]:
                self.assertIn("InertialCoordinates_" + component, observation)

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
        self.assertEqual(
            attributes["CharacteristicSpeedsOnStrahlkorper_t"], "Scalar"
        )
        self.assertEqual(
            attributes["CharacteristicSpeedsOnStrahlkorper"], "Vector"
        )


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
