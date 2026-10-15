#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

import argparse
import os
import unittest

import h5py
import numpy as np


class CheckPsi0RealVolumeOutput(unittest.TestCase):
    def test_psi0_and_psi4_are_pointwise_volume_data(self):
        volume_file = os.path.join(self.run_directory, "GhKerrSchildVolume0.h5")
        with h5py.File(volume_file, "r") as open_volume_file:
            observations = open_volume_file["VolumeData.vol"]
            observation_groups = [
                item
                for name, item in observations.items()
                if name.startswith("ObservationId")
            ]
            self.assertGreater(len(observation_groups), 0)
            for observation in observation_groups:
                self.assertIn("Psi0Real", observation)
                self.assertIn("Psi4Real", observation)
                psi0_real = observation["Psi0Real"][:]
                psi4_real = observation["Psi4Real"][:]
                self.assertEqual(psi0_real.shape, psi4_real.shape)
                self.assertTrue(np.all(np.isfinite(psi0_real)))
                self.assertTrue(np.all(np.isfinite(psi4_real)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-filename")
    parser.add_argument("--run-directory")
    parser.add_argument("--cmake-source-directory")
    parser.add_argument("--cmake-bin-directory")
    duplicate_test_case, remaining_args = parser.parse_known_args(
        namespace=CheckPsi0RealVolumeOutput
    )
    del duplicate_test_case
    unittest.main(argv=[parser.prog] + remaining_args, verbosity=2)
