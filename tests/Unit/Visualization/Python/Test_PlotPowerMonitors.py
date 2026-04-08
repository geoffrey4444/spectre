# Distributed under the MIT License.
# See LICENSE.txt for details.

import os
import shutil
import unittest

import numpy as np
from click.testing import CliRunner

from spectre.DataStructures import DataVector
from spectre.DataStructures.Tensor import Scalar
from spectre.Domain.Creators import Cylinder
from spectre.Informer import unit_test_build_path, unit_test_src_path
from spectre.NumericalAlgorithms.LinearOperators import (
    shell_power_monitor_buffer,
)
from spectre.Spectral import Basis, Mesh3D, Quadrature
from spectre.Visualization.PlotPowerMonitors import (
    _check_compatible_monitor_labels,
    _check_supported_shell_layout,
    _expand_symmetric_last_two_components,
    _select_tensor_components,
    _shell_buffers_for_tensor_variable,
    _tensor_variable_monitors,
    find_block_or_group,
    is_shell_mesh,
    plot_power_monitors_command,
)


class TestPlotPowerMonitors(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.join(
            unit_test_build_path(), "Visualization", "PlotPowerMonitors"
        )
        os.makedirs(self.test_dir, exist_ok=True)
        self.h5_filename = os.path.join(
            unit_test_src_path(), "Visualization/Python", "VolTestData0.h5"
        )
        self.plot_filename = os.path.join(self.test_dir, "plot.pdf")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_find_block_or_group(self):
        domain = Cylinder(
            inner_radius=1.0,
            outer_radius=3.0,
            lower_bound=0.0,
            upper_bound=2.0,
            is_periodic_in_z=False,
            initial_refinement=1,
            initial_number_of_grid_points=[3, 4, 5],
            use_equiangular_map=True,
        ).create_domain()
        self.assertEqual(
            find_block_or_group(0, ["BlockyBlock", "InnerCube"], domain), 1
        )
        self.assertEqual(
            find_block_or_group(1, ["BlockyBlock", "InnerCube"], domain), None
        )
        self.assertEqual(
            find_block_or_group(1, ["InnerCube", "Wedges"], domain), 1
        )

    def test_helpers(self):
        self.assertEqual(
            _select_tensor_components(["Psi", "Pi_tt", "Pi_tx", "Pi_xx"], "Pi"),
            ["Pi_tt", "Pi_tx", "Pi_xx"],
        )
        shell_mesh = Mesh3D(
            [3, 4, 7],
            [Basis.Legendre, Basis.SphericalHarmonic, Basis.SphericalHarmonic],
            [
                Quadrature.GaussLobatto,
                Quadrature.Equiangular,
                Quadrature.Equiangular,
            ],
        )
        self.assertTrue(is_shell_mesh(shell_mesh))
        non_radial_first_shell_mesh = Mesh3D(
            [4, 7, 3],
            [Basis.SphericalHarmonic, Basis.SphericalHarmonic, Basis.Legendre],
            [
                Quadrature.Gauss,
                Quadrature.Equiangular,
                Quadrature.GaussLobatto,
            ],
        )
        self.assertFalse(is_shell_mesh(non_radial_first_shell_mesh))
        with self.assertRaisesRegex(
            ValueError, "radial dimension first and the two spherical-harmonic"
        ):
            _check_supported_shell_layout(non_radial_first_shell_mesh)
        with self.assertRaisesRegex(
            ValueError, "mix spherical-shell elements with non-shell elements"
        ):
            _check_compatible_monitor_labels(
                ["radial", "angular-l"], [r"$\\xi$", r"$\\eta$", r"$\\zeta$"]
            )
        suffixes, data = _expand_symmetric_last_two_components(
            ["xx", "yx", "yy", "zx", "zy", "zz"],
            np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]),
        )
        self.assertEqual(
            suffixes,
            ["xx", "yx", "xy", "yy", "zx", "xz", "zy", "yz", "zz"],
        )
        np.testing.assert_allclose(
            data[:, 0], [1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0]
        )

    def test_shell_spacetime_tensor_includes_scalar_piece(self):
        shell_mesh = Mesh3D(
            [3, 4, 7],
            [Basis.Legendre, Basis.SphericalHarmonic, Basis.SphericalHarmonic],
            [
                Quadrature.GaussLobatto,
                Quadrature.Equiangular,
                Quadrature.Equiangular,
            ],
        )
        num_points = shell_mesh.number_of_grid_points()
        component_names = [
            "Psi_tt",
            "Psi_tx",
            "Psi_ty",
            "Psi_tz",
            "Psi_xx",
            "Psi_xy",
            "Psi_xz",
            "Psi_yy",
            "Psi_yz",
            "Psi_zz",
        ]
        tensor_data = [
            [float(i + 1)] * num_points for i in range(len(component_names))
        ]

        buffers = _shell_buffers_for_tensor_variable(
            component_names, tensor_data, shell_mesh
        )

        self.assertEqual(len(buffers), 3)
        np.testing.assert_allclose(
            np.array(buffers[0]["angular_counts"]),
            np.array(
                shell_power_monitor_buffer(
                    DataVector(np.array(tensor_data[0])), shell_mesh
                )["angular_counts"]
            ),
        )

        _, modes = _tensor_variable_monitors(
            component_names, tensor_data, shell_mesh, 0
        )
        expected_radial = np.sqrt(np.mean(np.square(tensor_data), axis=0)[0])
        self.assertAlmostEqual(modes[0][0], expected_radial)

    def test_shell_spacetime_tensor_sector_counts(self):
        shell_mesh = Mesh3D(
            [3, 4, 7],
            [Basis.Legendre, Basis.SphericalHarmonic, Basis.SphericalHarmonic],
            [
                Quadrature.GaussLobatto,
                Quadrature.Equiangular,
                Quadrature.Equiangular,
            ],
        )
        num_points = shell_mesh.number_of_grid_points()
        n_r = shell_mesh.extents(0)
        component_names = [
            "Psi_tt",
            "Psi_tx",
            "Psi_ty",
            "Psi_tz",
            "Psi_xx",
            "Psi_xy",
            "Psi_xz",
            "Psi_yy",
            "Psi_yz",
            "Psi_zz",
        ]
        tensor_data = [
            [float(i + 1)] * num_points for i in range(len(component_names))
        ]

        buffers = _shell_buffers_for_tensor_variable(
            component_names, tensor_data, shell_mesh
        )

        self.assertEqual(len(buffers), 3)
        np.testing.assert_array_equal(
            np.array(buffers[0]["angular_counts"]),
            n_r * np.array([1, 3, 5, 7]),
        )
        np.testing.assert_array_equal(
            np.array(buffers[1]["angular_counts"]),
            n_r * np.array([3, 12, 18, 24]),
        )
        np.testing.assert_array_equal(
            np.array(buffers[2]["angular_counts"]),
            n_r * np.array([6, 16, 36, 48]),
        )

    def test_cli(self):
        runner = CliRunner()
        # Test plotting a single step
        result = runner.invoke(
            plot_power_monitors_command,
            [
                self.h5_filename,
                "-d",
                "element_data",
                "--step",
                "-1",
                "-b",
                "Brick",
                "-e",
                "B*",
                "-y",
                "Psi",
                "--figsize",
                "12",
                "4",
                "-o",
                self.plot_filename,
            ],
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 0, result.output)
        # Can't easily test the plot itself, so just check that it was created
        self.assertTrue(os.path.exists(self.plot_filename))
        os.remove(self.plot_filename)

        result = runner.invoke(
            plot_power_monitors_command,
            [
                self.h5_filename,
                "-d",
                "element_data",
                "--step",
                "-1",
                "-b",
                "Brick",
                "-y",
                "Psi",
                "--tensor",
                "Psi",
            ],
            catch_exceptions=False,
        )
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("either '--var' / '-y' or '--tensor'", result.output)

        # Test plotting over time
        result = runner.invoke(
            plot_power_monitors_command,
            [
                self.h5_filename,
                "-d",
                "element_data",
                "-b",
                "Brick",
                "-e",
                "B*",
                "-y",
                "Psi",
                "--over-time",
                "-o",
                self.plot_filename,
            ],
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 0, result.output)
        # Can't easily test the plot itself, so just check that it was created
        self.assertTrue(os.path.exists(self.plot_filename))
        os.remove(self.plot_filename)


if __name__ == "__main__":
    unittest.main(verbosity=2)
