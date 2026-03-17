#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

import os
import shutil
import unittest

import h5py
import numpy as np
from click.testing import CliRunner

import spectre.Informer as spectre_informer
import spectre.IO.H5 as spectre_h5
from spectre.IO.H5.ConvertFiniteRadiusRwzToSpec import (
    convert_finite_radius_rwz_to_spec,
    convert_finite_radius_rwz_to_spec_command,
)


class TestConvertFiniteRadiusRwzToSpec(unittest.TestCase):
    def setUp(self):
        unit_test_build_dir = spectre_informer.unit_test_build_path()
        self.test_dir = os.path.join(
            unit_test_build_dir, "IO/H5/Python/TestConvertFiniteRadiusRwzToSpec"
        )
        self.native_file = os.path.join(self.test_dir, "NativeRwz.h5")
        self.output_dir = os.path.join(self.test_dir, "GW2")
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir, exist_ok=True)

        with spectre_h5.H5File(file_name=self.native_file, mode="r+") as h5file:
            for quantity, legend, data in (
                (
                    "/FiniteRadiusExtraction/Radius24_0/CoordRadius",
                    ["Time", "CoordRadius"],
                    np.array([[0.0, 24.0], [1.0, 24.0]]),
                ),
                (
                    "/FiniteRadiusExtraction/Radius24_0/InitialAdmEnergy",
                    ["Time", "InitialAdmEnergy"],
                    np.array([[0.0, 0.98], [1.0, 0.98]]),
                ),
                (
                    "/FiniteRadiusExtraction/Radius24_0/AverageLapse",
                    ["Time", "AverageLapse"],
                    np.array([[0.0, 0.9], [1.0, 0.8]]),
                ),
                (
                    "/FiniteRadiusExtraction/Radius24_0/ArealRadius",
                    ["Time", "ArealRadius"],
                    np.array([[0.0, 23.9], [1.0, 23.8]]),
                ),
                (
                    "/FiniteRadiusExtraction/Radius24_0/Strain",
                    [
                        "Time",
                        "Re(0,0)",
                        "Im(0,0)",
                        "Re(2,-2)",
                        "Im(2,-2)",
                        "Re(2,-1)",
                        "Im(2,-1)",
                        "Re(2,0)",
                        "Im(2,0)",
                        "Re(2,1)",
                        "Im(2,1)",
                        "Re(2,2)",
                        "Im(2,2)",
                    ],
                    np.array(
                        [
                            [
                                0.0,
                                0.0,
                                0.0,
                                1.0,
                                -1.0,
                                2.0,
                                -2.0,
                                3.0,
                                -3.0,
                                4.0,
                                -4.0,
                                5.0,
                                -5.0,
                            ],
                            [
                                1.0,
                                0.0,
                                0.0,
                                1.5,
                                -1.5,
                                2.5,
                                -2.5,
                                3.5,
                                -3.5,
                                4.5,
                                -4.5,
                                5.5,
                                -5.5,
                            ],
                        ]
                    ),
                ),
                (
                    "/FiniteRadiusExtraction/Radius24_0/PhiPlus",
                    [
                        "Time",
                        "Re(0,0)",
                        "Im(0,0)",
                        "Re(2,-2)",
                        "Im(2,-2)",
                        "Re(2,-1)",
                        "Im(2,-1)",
                        "Re(2,0)",
                        "Im(2,0)",
                        "Re(2,1)",
                        "Im(2,1)",
                        "Re(2,2)",
                        "Im(2,2)",
                    ],
                    np.array(
                        [
                            [
                                0.0,
                                0.0,
                                0.0,
                                0.1,
                                -0.1,
                                0.2,
                                -0.2,
                                0.3,
                                -0.3,
                                0.4,
                                -0.4,
                                0.5,
                                -0.5,
                            ],
                            [
                                1.0,
                                0.0,
                                0.0,
                                0.15,
                                -0.15,
                                0.25,
                                -0.25,
                                0.35,
                                -0.35,
                                0.45,
                                -0.45,
                                0.55,
                                -0.55,
                            ],
                        ]
                    ),
                ),
                (
                    "/FiniteRadiusExtraction/Radius24_0/PhiMinus",
                    [
                        "Time",
                        "Re(0,0)",
                        "Im(0,0)",
                        "Re(2,-2)",
                        "Im(2,-2)",
                        "Re(2,-1)",
                        "Im(2,-1)",
                        "Re(2,0)",
                        "Im(2,0)",
                        "Re(2,1)",
                        "Im(2,1)",
                        "Re(2,2)",
                        "Im(2,2)",
                    ],
                    np.array(
                        [
                            [
                                0.0,
                                0.0,
                                0.0,
                                -0.1,
                                0.1,
                                -0.2,
                                0.2,
                                -0.3,
                                0.3,
                                -0.4,
                                0.4,
                                -0.5,
                                0.5,
                            ],
                            [
                                1.0,
                                0.0,
                                0.0,
                                -0.15,
                                0.15,
                                -0.25,
                                0.25,
                                -0.35,
                                0.35,
                                -0.45,
                                0.45,
                                -0.55,
                                0.55,
                            ],
                        ]
                    ),
                ),
                (
                    "/FiniteRadiusExtraction/Radius24_0/Psi4",
                    [
                        "Time",
                        "Re(0,0)",
                        "Im(0,0)",
                        "Re(2,-2)",
                        "Im(2,-2)",
                        "Re(2,-1)",
                        "Im(2,-1)",
                        "Re(2,0)",
                        "Im(2,0)",
                        "Re(2,1)",
                        "Im(2,1)",
                        "Re(2,2)",
                        "Im(2,2)",
                    ],
                    np.array(
                        [
                            [
                                0.0,
                                0.0,
                                0.0,
                                0.6,
                                -0.6,
                                0.7,
                                -0.7,
                                0.8,
                                -0.8,
                                0.9,
                                -0.9,
                                1.0,
                                -1.0,
                            ],
                            [
                                1.0,
                                0.0,
                                0.0,
                                0.65,
                                -0.65,
                                0.75,
                                -0.75,
                                0.85,
                                -0.85,
                                0.95,
                                -0.95,
                                1.05,
                                -1.05,
                            ],
                        ]
                    ),
                ),
            ):
                dat_file = h5file.insert_dat(
                    path=quantity, legend=legend, version=0
                )
                dat_file.append(data)
                h5file.close_current_object()

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_convert_finite_radius_rwz_to_spec(self):
        convert_finite_radius_rwz_to_spec(self.native_file, self.output_dir)

        output_files = {
            "rh": os.path.join(self.output_dir, "rh_FiniteRadii_CodeUnits.h5"),
            "PhiPlus": os.path.join(
                self.output_dir, "PhiPlus_FiniteRadii_CodeUnits.h5"
            ),
            "PhiMinus": os.path.join(
                self.output_dir, "PhiMinus_FiniteRadii_CodeUnits.h5"
            ),
            "Psi4": os.path.join(
                self.output_dir, "rPsi4_FiniteRadii_CodeUnits.h5"
            ),
        }
        for output_file in output_files.values():
            self.assertTrue(os.path.exists(output_file))

        with h5py.File(output_files["rh"], "r") as h5file:
            self.assertIn("R0024.dir/CoordRadius.dat", h5file)
            self.assertIn("R0024.dir/InitialAdmEnergy.dat", h5file)
            self.assertIn("R0024.dir/AverageLapse.dat", h5file)
            self.assertIn("R0024.dir/ArealRadius.dat", h5file)
            self.assertIn("R0024.dir/Y_l2_m2.dat", h5file)
            self.assertNotIn("R0024.dir/Y_l0_m0.dat", h5file)
            np.testing.assert_allclose(
                h5file["R0024.dir/CoordRadius.dat"], [[0.0, 24.0], [1.0, 24.0]]
            )
            np.testing.assert_allclose(
                h5file["R0024.dir/InitialAdmEnergy.dat"], [[0.0, 0.98]]
            )
            self.assertEqual(
                list(h5file["R0024.dir/Y_l2_m2.dat"].attrs["Legend"]),
                [
                    "time",
                    "Re[rh]_l2_m2(R=24)",
                    "Im[rh]_l2_m2(R=24)",
                ],
            )
            np.testing.assert_allclose(
                h5file["R0024.dir/Y_l2_m2.dat"],
                [[0.0, 5.0, -5.0], [1.0, 5.5, -5.5]],
            )

        with h5py.File(output_files["Psi4"], "r") as h5file:
            self.assertIn("R0024.dir/Y_l2_m2.dat", h5file)
            self.assertEqual(
                list(h5file["R0024.dir/Y_l2_m2.dat"].attrs["Legend"]),
                [
                    "time",
                    "Re[rPsi4]_l2_m2(R=24)",
                    "Im[rPsi4]_l2_m2(R=24)",
                ],
            )
            np.testing.assert_allclose(
                h5file["R0024.dir/Y_l2_m2.dat"],
                [[0.0, 1.0, -1.0], [1.0, 1.05, -1.05]],
            )

    def test_cli(self):
        runner = CliRunner()
        result = runner.invoke(
            convert_finite_radius_rwz_to_spec_command,
            [self.native_file, "--output-dir", self.output_dir],
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
