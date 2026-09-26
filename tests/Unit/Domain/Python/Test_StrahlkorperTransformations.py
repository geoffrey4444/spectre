# Distributed under the MIT License.
# See LICENSE.txt for details.

import os
import unittest

import numpy as np
import numpy.testing as npt

import spectre.IO.H5 as spectre_h5
from spectre.DataStructures import DataVector
from spectre.Domain import (
    PiecewisePolynomial2,
    deserialize_domain,
    deserialize_functions_of_time,
    strahlkorper_in_distorted_frame,
    strahlkorper_in_inertial_frame,
    strahlkorper_in_inertial_frame_aligned,
)
from spectre.Domain.Creators import BinaryCompactObject, Sphere
from spectre.Domain.Creators.TimeDependentOptions import (
    BinaryCompactObjectTimeDependentOptions,
    ShapeMapOptions,
    TranslationMapOptions,
)
from spectre.Informer import unit_test_src_path
from spectre.Strahlkorper import Frame, Strahlkorper, cartesian_coords


class TestStrahlkorperTransformations(unittest.TestCase):
    def test_moving_distorted_frame(self):
        creator = BinaryCompactObject(
            inner_radius_a=0.5,
            outer_radius_a=2.0,
            x_coord_a=5.0,
            excise_a=True,
            use_logarithmic_map_a=True,
            inner_radius_b=0.5,
            outer_radius_b=2.0,
            x_coord_b=-5.0,
            excise_b=True,
            use_logarithmic_map_b=True,
            center_of_mass_offset=[0.0, 0.0],
            envelope_radius=50.0,
            outer_radius=100.0,
            cube_scale=1.2,
            initial_refinement=0,
            initial_number_of_grid_points=4,
            use_equiangular_map=True,
            time_dependent_options=BinaryCompactObjectTimeDependentOptions(
                initial_time=0.0,
                expansion_map_options=None,
                rotation_map_options=None,
                translation_map_options=TranslationMapOptions(
                    [[0.2, -0.1, 0.3], [0.05, 0.0, -0.02], [0.0] * 3]
                ),
                skew_map_options=None,
                shape_options_A=ShapeMapOptions["A"](4, None),
                shape_options_B=None,
                grid_centers_options=None,
            ),
        )
        domain = creator.create_domain()
        surface = Strahlkorper[Frame.Inertial](
            l_max=4, radius=1.0, center=[5.25, -0.1, 0.28]
        )
        transformed = strahlkorper_in_distorted_frame(
            surface, domain, creator.functions_of_time(), 1.0
        )
        self.assertIsInstance(transformed, Strahlkorper[Frame.Distorted])
        npt.assert_allclose(
            transformed.expansion_center, [5.0, 0.0, 0.0], atol=1.0e-14
        )
        npt.assert_allclose(
            np.linalg.norm(
                np.asarray(cartesian_coords(transformed))
                - np.array([[5.0], [0.0], [0.0]]),
                axis=0,
            ),
            1.0,
            atol=1.0e-14,
        )

    def test_static_distorted_frame(self):
        domain = Sphere(
            inner_radius=1.0,
            outer_radius=5.0,
            initial_refinement=0,
            initial_number_of_grid_points=4,
            use_equiangular_map=True,
            excise=True,
        ).create_domain()
        surface = Strahlkorper[Frame.Inertial](
            l_max=5, radius=2.0, center=[0.1, -0.2, 0.3]
        )
        transformed = strahlkorper_in_distorted_frame(surface, domain)
        self.assertIsInstance(transformed, Strahlkorper[Frame.Distorted])
        self.assertEqual(transformed.expansion_center, surface.expansion_center)
        npt.assert_allclose(
            cartesian_coords(transformed), cartesian_coords(surface)
        )

    def test_strahlkorper_in_different_frame(self):
        volfile_name = os.path.join(
            unit_test_src_path(), "Visualization/Python/VolTestData0.h5"
        )
        with spectre_h5.H5File(volfile_name, "r") as open_h5_file:
            volfile = open_h5_file.get_vol("/element_data")
            obs_id = volfile.list_observation_ids()[0]
            domain = deserialize_domain[3](volfile.get_domain())
            functions_of_time = deserialize_functions_of_time(
                volfile.get_functions_of_time(obs_id)
            )

        strahlkorper_grid = Strahlkorper[Frame.Grid](
            l_max=2, radius=0.5, center=[0.5, 0.5, 0.5]
        )
        strahlkorper_inertial = strahlkorper_in_inertial_frame(
            strahlkorper_grid,
            domain=domain,
            functions_of_time=functions_of_time,
            time=0.0,
        )
        self.assertAlmostEqual(strahlkorper_inertial.average_radius, 0.5)
        with self.assertRaisesRegex(ValueError, "finite time"):
            strahlkorper_in_distorted_frame(
                strahlkorper_inertial,
                domain=domain,
                functions_of_time=functions_of_time,
            )
        with self.assertRaisesRegex(ValueError, "Translation.*missing"):
            strahlkorper_in_distorted_frame(
                strahlkorper_inertial, domain=domain, time=0.0
            )
        expired_functions_of_time = {
            "Translation": PiecewisePolynomial2(
                time=0.0,
                initial_func_and_derivs=[DataVector([0.0] * 3)] * 3,
                expiration_time=0.5,
            )
        }
        for time in [-1.0, 1.0]:
            with self.subTest(time=time):
                with self.assertRaisesRegex(ValueError, "Translation.*bounds"):
                    strahlkorper_in_distorted_frame(
                        strahlkorper_inertial,
                        domain=domain,
                        functions_of_time=expired_functions_of_time,
                        time=time,
                    )
        with self.assertRaisesRegex(ValueError, "distorted frame"):
            strahlkorper_in_distorted_frame(
                strahlkorper_inertial,
                domain=domain,
                functions_of_time=functions_of_time,
                time=0.0,
            )
        strahlkorper_inertial = strahlkorper_in_inertial_frame_aligned(
            strahlkorper_grid,
            domain=domain,
            functions_of_time=functions_of_time,
            time=0.0,
        )
        self.assertAlmostEqual(strahlkorper_inertial.average_radius, 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
