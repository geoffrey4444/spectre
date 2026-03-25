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
from spectre.IO.H5.VerifyFiniteRadiusStrainPsi4 import (
    verify_finite_radius_strain_psi4,
    verify_finite_radius_strain_psi4_command,
)


def second_derivative_samples(time, values):
    dt_plus = time[2:] - time[1:-1]
    dt_minus = time[1:-1] - time[:-2]
    return 2.0 * (
        values[:-2] / (dt_minus * (dt_plus + dt_minus))
        - values[1:-1] / (dt_plus * dt_minus)
        + values[2:] / (dt_plus * (dt_plus + dt_minus))
    )


class TestVerifyFiniteRadiusStrainPsi4(unittest.TestCase):
    def setUp(self):
        unit_test_build_dir = spectre_informer.unit_test_build_path()
        self.test_dir = os.path.join(
            unit_test_build_dir, "IO/H5/Python/TestVerifyFiniteRadiusStrainPsi4"
        )
        self.strain_file = os.path.join(
            self.test_dir, "rh_FiniteRadii_CodeUnits.h5"
        )
        self.psi4_file = os.path.join(
            self.test_dir, "rPsi4_FiniteRadii_CodeUnits.h5"
        )
        self.bad_psi4_file = os.path.join(
            self.test_dir, "rPsi4_FiniteRadii_CodeUnits_Bad.h5"
        )
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir, exist_ok=True)

        time = np.array([0.0, 0.7, 1.8, 3.0, 4.5])
        strain_22_real = time**2 + 2.0 * time + 1.0
        strain_22_imag = -0.5 * time**2 + time - 3.0
        strain_21_real = -2.0 * time**2 + 0.5 * time
        strain_21_imag = 0.25 * time**2 - 2.0 * time + 1.0

        psi4_22_real = -second_derivative_samples(time, strain_22_real)
        psi4_22_imag = -second_derivative_samples(time, strain_22_imag)
        psi4_21_real = -second_derivative_samples(time, strain_21_real)
        psi4_21_imag = -second_derivative_samples(time, strain_21_imag)

        strain_22_data = np.column_stack((time, strain_22_real, strain_22_imag))
        strain_21_data = np.column_stack((time, strain_21_real, strain_21_imag))
        psi4_22_data = np.column_stack((time[1:-1], psi4_22_real, psi4_22_imag))
        psi4_21_data = np.column_stack((time[1:-1], psi4_21_real, psi4_21_imag))
        bad_psi4_22_data = psi4_22_data.copy()
        bad_psi4_22_data[0, 1] += 0.25

        self.write_waveform_file(
            self.strain_file,
            {
                "/R0012.dir/Y_l2_m1": strain_21_data,
                "/R0012.dir/Y_l2_m2": strain_22_data,
                "/R0024.dir/Y_l2_m1": strain_21_data,
                "/R0024.dir/Y_l2_m2": strain_22_data,
            },
            quantity_name="rh",
        )
        self.write_waveform_file(
            self.psi4_file,
            {
                "/R0012.dir/Y_l2_m1": psi4_21_data,
                "/R0012.dir/Y_l2_m2": psi4_22_data,
                "/R0024.dir/Y_l2_m1": psi4_21_data,
                "/R0024.dir/Y_l2_m2": psi4_22_data,
            },
            quantity_name="rPsi4",
        )
        self.write_waveform_file(
            self.bad_psi4_file,
            {
                "/R0012.dir/Y_l2_m1": psi4_21_data,
                "/R0012.dir/Y_l2_m2": psi4_22_data,
                "/R0024.dir/Y_l2_m1": psi4_21_data,
                "/R0024.dir/Y_l2_m2": bad_psi4_22_data,
            },
            quantity_name="rPsi4",
        )

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @staticmethod
    def write_waveform_file(file_name, datasets, quantity_name):
        with spectre_h5.H5File(file_name=file_name, mode="r+") as h5file:
            for subfile, data in datasets.items():
                dataset_name = subfile.split("/")[-1]
                match = dataset_name.removesuffix(".dat").split("_")
                l = match[1][1:]
                m = match[2][1:]
                dat_file = h5file.insert_dat(
                    path=subfile,
                    legend=[
                        "time",
                        f"Re[{quantity_name}]_l{l}_m{m}(R=0)",
                        f"Im[{quantity_name}]_l{l}_m{m}(R=0)",
                    ],
                    version=0,
                )
                dat_file.append(data)
                h5file.close_current_object()

    def test_verify_finite_radius_strain_psi4(self):
        verify_finite_radius_strain_psi4(self.strain_file, self.psi4_file)
        with self.assertRaises(AssertionError):
            verify_finite_radius_strain_psi4(
                self.strain_file, self.bad_psi4_file
            )

    def test_cli(self):
        runner = CliRunner()
        result = runner.invoke(
            verify_finite_radius_strain_psi4_command,
            [self.strain_file, self.psi4_file],
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
