#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

import os
import shutil
import unittest

import numpy as np
from click.testing import CliRunner

import spectre.Informer as spectre_informer
import spectre.IO.H5 as spectre_h5
from spectre.IO.H5.CompareFiniteRadiusRwz import (
    compare_finite_radius_rwz,
    compare_finite_radius_rwz_command,
)


class TestCompareFiniteRadiusRwz(unittest.TestCase):
    def setUp(self):
        unit_test_build_dir = spectre_informer.unit_test_build_path()
        self.test_dir = os.path.join(
            unit_test_build_dir, "IO/H5/Python/TestCompareFiniteRadiusRwz"
        )
        self.reference_file = os.path.join(self.test_dir, "Reference.h5")
        self.matching_file = os.path.join(self.test_dir, "Matching.h5")
        self.mismatched_file = os.path.join(self.test_dir, "Mismatched.h5")
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir, exist_ok=True)

        self.write_rwz_file(
            self.reference_file,
            mode_data=np.array([[0.0, 1.0, -1.0], [1.0, 2.0, -2.0]]),
        )
        self.write_rwz_file(
            self.matching_file,
            mode_data=np.array([[0.0, 1.0, -1.0], [1.0, 2.0, -2.0]]),
        )
        self.write_rwz_file(
            self.mismatched_file,
            mode_data=np.array([[0.0, 1.5, -1.0], [1.0, 2.0, -2.0]]),
        )

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @staticmethod
    def write_rwz_file(file_name, mode_data):
        with spectre_h5.H5File(file_name=file_name, mode="r+") as h5file:
            for subfile, legend, data in (
                (
                    "/R0024.dir/CoordRadius",
                    ["time", "CoordRadius"],
                    np.array([[0.0, 24.0], [1.0, 24.0]]),
                ),
                (
                    "/R0024.dir/Y_l2_m2",
                    ["time", "Re[rh]_l2_m2(R=24)", "Im[rh]_l2_m2(R=24)"],
                    mode_data,
                ),
            ):
                dat_file = h5file.insert_dat(
                    path=subfile, legend=legend, version=0
                )
                dat_file.append(data)
                h5file.close_current_object()

    def test_compare_finite_radius_rwz(self):
        compare_finite_radius_rwz(self.reference_file, self.matching_file)
        with self.assertRaises(AssertionError):
            compare_finite_radius_rwz(self.reference_file, self.mismatched_file)

    def test_cli(self):
        runner = CliRunner()
        result = runner.invoke(
            compare_finite_radius_rwz_command,
            [self.reference_file, self.matching_file],
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
