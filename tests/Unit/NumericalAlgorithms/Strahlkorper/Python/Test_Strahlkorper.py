# Distributed under the MIT License.
# See LICENSE.txt for details.

import os
import shutil
import unittest

import numpy as np
import numpy.testing as npt

import spectre.Informer as spectre_informer
import spectre.IO.H5 as spectre_h5
from spectre.DataStructures import ModalVector
from spectre.Strahlkorper import (
    AngularOrdering,
    Frame,
    Strahlkorper,
    cartesian_coords,
    change_expansion_center_of_strahlkorper,
    power_monitor,
    read_surface_ylm,
    read_surface_ylm_distorted,
    read_surface_ylm_single_time,
    read_surface_ylm_single_time_distorted,
    time_deriv_of_strahlkorper,
    write_sphere_of_points_to_text_file,
    ylm_legend_and_data,
)


class TestStrahlkorper(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.join(
            spectre_informer.unit_test_build_path(),
            "NumericalAlgorithms/Strahlkorper/Python",
        )
        self.filename = os.path.join(self.test_dir, "Strahlkorper.h5")
        self.text_filename = os.path.join(
            self.test_dir, "PyStrahlkorperCoords.txt"
        )
        shutil.rmtree(self.test_dir, ignore_errors=True)
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_strahlkorper(self):
        strahlkorper = Strahlkorper[Frame.Inertial](
            l_max=12, radius=1.0, center=[0.0, 0.0, 0.0]
        )
        self.assertEqual(strahlkorper.l_max, 12)
        self.assertEqual(strahlkorper.m_max, 12)
        self.assertEqual(strahlkorper.physical_extents, [13, 25])
        self.assertEqual(strahlkorper.expansion_center, [0.0, 0.0, 0.0])
        self.assertEqual(strahlkorper.physical_center, [0.0, 0.0, 0.0])
        self.assertAlmostEqual(strahlkorper.average_radius, 1.0)
        self.assertAlmostEqual(strahlkorper.radius(0.0, 0.0), 1.0)
        self.assertTrue(strahlkorper.point_is_contained([0.5, 0.0, 0.0]))
        x = np.array(cartesian_coords(strahlkorper))
        r = np.linalg.norm(x, axis=0)
        npt.assert_allclose(r, 1.0)

        power = np.array(power_monitor(strahlkorper))
        for i, power_in_mode in enumerate(power):
            if i > 0:
                self.assertAlmostEqual(power_in_mode, 0.0)
            else:
                self.assertAlmostEqual(power_in_mode, 2.0 * np.sqrt(2.0))

        legend, ylm_data = ylm_legend_and_data(strahlkorper, 1.0, 12)
        self.assertEqual(len(legend), 174)
        self.assertEqual(
            legend[:7],
            [
                "Time",
                "InertialExpansionCenter_x",
                "InertialExpansionCenter_y",
                "InertialExpansionCenter_z",
                "Lmax",
                "coef(0,0)",
                "coef(1,-1)",
            ],
        )
        self.assertEqual(ylm_data[:5], [1.0, 0.0, 0.0, 0.0, 12.0])

        with spectre_h5.H5File(self.filename, "w") as h5file:
            datfile = h5file.insert_dat(
                "/Strahlkorper", legend=legend, version=0
            )
            datfile.append(ylm_data)

        self.assertEqual(
            read_surface_ylm(self.filename, "Strahlkorper", 1)[0], strahlkorper
        )
        self.assertEqual(
            read_surface_ylm_single_time(
                self.filename, "Strahlkorper", 1.0, 0.0, True
            ),
            strahlkorper,
        )

        l_max = 4
        print("hello")
        # First write with wrong l_max
        write_sphere_of_points_to_text_file(
            radius=1.2,
            l_max=l_max - 1,
            center=[-0.1, -0.2, -0.3],
            output_file_name=self.text_filename,
            ordering=AngularOrdering.Cce,
        )
        # Test that if overwrite_file = False (the default) an exception is
        # raised
        self.assertRaises(
            RuntimeError,
            write_sphere_of_points_to_text_file,
            radius=1.2,
            l_max=l_max - 1,
            center=[-0.1, -0.2, -0.3],
            output_file_name=self.text_filename,
            ordering=AngularOrdering.Cce,
        )
        # Finally write the correct l_max so we can check that overwrite_file
        # works
        write_sphere_of_points_to_text_file(
            radius=1.2,
            l_max=l_max,
            center=[-0.1, -0.2, -0.3],
            output_file_name=self.text_filename,
            ordering=AngularOrdering.Cce,
            overwrite_file=True,
        )

        with open(self.text_filename, "r") as text_file:
            # Physical size of ylm::Spherepack
            num_points = (l_max + 1) * (2 * l_max + 1)
            all_lines = text_file.readlines()
            self.assertEqual(num_points, len(all_lines))

    def test_distorted_surface_io(self):
        surfaces = [
            Strahlkorper[Frame.Distorted](
                l_max=4, radius=radius, center=[0.1, -0.2, 0.3]
            )
            for radius in [2.0, 2.1]
        ]
        with spectre_h5.H5File(self.filename, "w") as h5file:
            legend, data = ylm_legend_and_data(surfaces[0], 1.0, 4)
            datfile = h5file.insert_dat("Surface", legend=legend, version=0)
            datfile.append(data)
            datfile.append(ylm_legend_and_data(surfaces[1], 2.0, 4)[1])
            h5file.close_current_object()
            inertial_surface = Strahlkorper[Frame.Inertial](
                l_max=4, radius=2.0, center=[0.1, -0.2, 0.3]
            )
            legend, data = ylm_legend_and_data(inertial_surface, 1.0, 4)
            datfile = h5file.insert_dat(
                "InertialSurface", legend=legend, version=0
            )
            datfile.append(data)
        self.assertEqual(
            read_surface_ylm_distorted(self.filename, "Surface", 2), surfaces
        )
        self.assertEqual(
            read_surface_ylm_single_time_distorted(
                self.filename, "Surface", 2.0, 1.0e-12, True
            ),
            surfaces[1],
        )
        with self.assertRaisesRegex(RuntimeError, "InertialExpansionCenter"):
            read_surface_ylm_single_time(
                self.filename, "Surface", 2.0, 0.0, True
            )
        with self.assertRaisesRegex(RuntimeError, "DistortedExpansionCenter"):
            read_surface_ylm_single_time_distorted(
                self.filename, "InertialSurface", 1.0, 0.0
            )
        with self.assertRaisesRegex(RuntimeError, "DistortedExpansionCenter"):
            read_surface_ylm_distorted(self.filename, "InertialSurface", 1)
        surface = surfaces[0]
        reconstructed = Strahlkorper[Frame.Distorted](
            surface.l_max,
            surface.m_max,
            ModalVector(np.asarray(surface.coefficients)),
            surface.expansion_center,
        )
        self.assertEqual(reconstructed, surface)
        coefficients = surface.coefficients
        coefficients[0] = 0.0
        self.assertEqual(surface, reconstructed)
        coords = np.asarray(cartesian_coords(surface))
        npt.assert_allclose(
            np.linalg.norm(coords - np.array([[0.1], [-0.2], [0.3]]), axis=0),
            2.0,
        )

    def test_surface_history_and_center(self):
        for frame in [Frame.Inertial, Frame.Grid, Frame.Distorted]:
            with self.subTest(frame=frame):
                history = [
                    (
                        time,
                        Strahlkorper[frame](
                            l_max=l_max,
                            radius=2.0 + 0.1 * time**2,
                            center=[0.0, 0.0, 0.0],
                        ),
                    )
                    for time, l_max in [(3.0, 8), (1.5, 6), (0.0, 4)]
                ]
                derivative = time_deriv_of_strahlkorper(history)
                self.assertEqual(derivative.l_max, 8)
                self.assertAlmostEqual(derivative.average_radius, 0.6)
                self.assertAlmostEqual(
                    time_deriv_of_strahlkorper(history[:1]).average_radius,
                    0.0,
                )
                with self.assertRaisesRegex(ValueError, "newest first"):
                    time_deriv_of_strahlkorper(history[::-1])
                surface = history[0][1]
                recentered = change_expansion_center_of_strahlkorper(
                    surface, [0.01, -0.02, 0.03]
                )
                self.assertEqual(surface.expansion_center, [0.0, 0.0, 0.0])
                self.assertEqual(
                    recentered.expansion_center, [0.01, -0.02, 0.03]
                )
                npt.assert_allclose(
                    np.linalg.norm(cartesian_coords(recentered), axis=0),
                    surface.average_radius,
                    atol=1.0e-12,
                )
        with self.assertRaisesRegex(ValueError, "one to four"):
            time_deriv_of_strahlkorper([])

    def test_history_with_different_m_max(self):
        for m_max in [2, 4]:
            with self.subTest(m_max=m_max):
                newest = Strahlkorper[Frame.Distorted](
                    4,
                    m_max,
                    Strahlkorper[Frame.Distorted](4, 2.1, [0.0] * 3),
                )
                older = Strahlkorper[Frame.Distorted](
                    4,
                    6 - m_max,
                    Strahlkorper[Frame.Distorted](4, 2.0, [0.0] * 3),
                )
                derivative = time_deriv_of_strahlkorper(
                    [(1.0, newest), (0.0, older)]
                )
                self.assertEqual(derivative.l_max, newest.l_max)
                self.assertEqual(derivative.m_max, newest.m_max)
                self.assertAlmostEqual(derivative.average_radius, 0.1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
