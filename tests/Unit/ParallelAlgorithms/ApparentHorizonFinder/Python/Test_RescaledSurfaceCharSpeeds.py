# Distributed under the MIT License.
# See LICENSE.txt for details.

import unittest

import numpy as np
import numpy.testing as npt

from spectre.ApparentHorizonFinder import (
    prepare_horizon_for_char_speeds,
    rescaled_surface_char_speed_extrema,
    rescaled_surface_factors,
    sample_rescaled_surface_char_speeds,
)
from spectre.DataStructures import DataVector
from spectre.DataStructures.Tensor import Scalar, tnsr
from spectre.Domain import ElementId
from spectre.Domain.Creators import Sphere
from spectre.Spectral import Basis, Mesh, Quadrature
from spectre.Strahlkorper import Frame, Strahlkorper, time_deriv_of_strahlkorper


class TestRescaledSurfaceCharSpeeds(unittest.TestCase):
    def test_kernels_and_sampling(self):
        domain = Sphere(
            inner_radius=1.0,
            outer_radius=3.0,
            excise=True,
            initial_refinement=0,
            initial_number_of_grid_points=4,
            use_equiangular_map=True,
        ).create_domain()
        surface = Strahlkorper[Frame.Distorted](4, 2.0, [0.0, 0.0, 0.0])
        derivative = Strahlkorper[Frame.Distorted](4, 0.2, [0.0, 0.0, 0.0])
        size = np.prod(surface.physical_extents)
        npt.assert_allclose(
            rescaled_surface_factors(
                DataVector(size, 2.0), DataVector(size, 1.0), 3, 0.0
            ),
            [1.0, 0.875, 0.5],
        )
        inverse_metric = tnsr.II[DataVector, 3, Frame.Distorted](size, 0.0)
        for i in range(3):
            inverse_metric[inverse_metric.get_storage_index(i, i)] = DataVector(
                size, 1.0
            )
        npt.assert_allclose(
            rescaled_surface_char_speed_extrema(
                surface,
                derivative,
                Scalar[DataVector](size, 1.0),
                tnsr.I[DataVector, 3, Frame.Distorted](size, 0.0),
                inverse_metric,
            ),
            [-0.8, -0.8],
        )
        self.assertEqual(
            prepare_horizon_for_char_speeds(surface, domain, "ExcisionSphere"),
            surface,
        )
        with self.assertRaisesRegex(ValueError, "excision"):
            prepare_horizon_for_char_speeds(surface, domain, "Absent")
        history = [
            (
                time,
                prepare_horizon_for_char_speeds(
                    Strahlkorper[Frame.Distorted](
                        4, 2.0, [1.0e-17 * (time + 1.0), 0.0, 0.0]
                    ),
                    domain,
                    "ExcisionSphere",
                ),
            )
            for time in [1.0, 0.0]
        ]
        for _, horizon in history:
            npt.assert_array_equal(horizon.expansion_center, [0.0, 0.0, 0.0])
        npt.assert_allclose(
            time_deriv_of_strahlkorper(history).coefficients, 0.0, atol=1.0e-14
        )
        mesh = Mesh[3](4, Basis.Legendre, Quadrature.GaussLobatto)
        metric = tnsr.aa[DataVector, 3](mesh.number_of_grid_points(), 0.0)
        for i in range(4):
            metric[metric.get_storage_index(i, i)] = DataVector(
                mesh.number_of_grid_points(), -1.0 if i == 0 else 1.0
            )
        legend, row = sample_rescaled_surface_char_speeds(
            surface,
            derivative,
            [ElementId[3](i) for i in range(6)],
            [mesh] * 6,
            [metric] * 6,
            domain,
            {},
            0.0,
            "ExcisionSphere",
            number_of_surfaces=3,
        )
        self.assertEqual(legend[:3], ["Time", "Status", "RadiusFactor_0"])
        self.assertEqual(row[1], 0.0)
        npt.assert_allclose(
            np.array(row[3::3]), -1.0 + 0.2 * np.array(row[2::3])
        )
        npt.assert_allclose(row[3::3], row[4::3])


if __name__ == "__main__":
    unittest.main(verbosity=2)
