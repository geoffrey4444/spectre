# Distributed under the MIT License.
# See LICENSE.txt for details.

import unittest

import numpy as np
import numpy.testing as npt

from spectre.DataStructures import DataVector
from spectre.DataStructures.Tensor import tnsr
from spectre.Domain import ElementId, ElementMap
from spectre.Domain.Creators import Sphere
from spectre.IO.Exporter import ObservationId
from spectre.Pipelines.Bbh.FindHorizon import find_horizon
from spectre.Pipelines.Bbh.HorizonInterpolation import GhHorizonInterpolator
from spectre.PointwiseFunctions.AnalyticSolutions.GeneralRelativity import (
    KerrSchild,
)
from spectre.PointwiseFunctions.GeneralRelativity import spacetime_metric
from spectre.PointwiseFunctions.GeneralRelativity.GeneralizedHarmonic import (
    phi,
    pi,
)
from spectre.Spectral import Basis, Mesh, Quadrature, logical_coordinates
from spectre.Strahlkorper import Frame, Strahlkorper, cartesian_coords


class TestHorizonInterpolation(unittest.TestCase):
    def setUp(self):
        self.domain = Sphere(
            inner_radius=1.0,
            outer_radius=3.0,
            excise=True,
            initial_refinement=0,
            initial_number_of_grid_points=10,
            use_equiangular_map=True,
        ).create_domain()
        self.element_ids = [ElementId[3](i) for i in range(6)]
        self.mesh = Mesh[3](12, Basis.Legendre, Quadrature.GaussLobatto)
        self.solution = KerrSchild(1.0, [0.0, 0.0, 0.0])
        spacetime_metrics, pis, phis = [], [], []
        for element_id in self.element_ids:
            x = ElementMap(element_id, self.domain)(
                logical_coordinates(self.mesh), None, None
            )
            fields = self.solution.variables(
                x,
                [
                    "Lapse",
                    "dt(Lapse)",
                    "deriv(Lapse)",
                    "Shift",
                    "dt(Shift)",
                    "deriv(Shift)",
                    "SpatialMetric",
                    "dt(SpatialMetric)",
                    "deriv(SpatialMetric)",
                ],
            )
            spacetime_metrics.append(
                spacetime_metric(
                    fields["Lapse"], fields["Shift"], fields["SpatialMetric"]
                )
            )
            phis.append(
                phi(
                    fields["Lapse"],
                    fields["deriv(Lapse)"],
                    fields["Shift"],
                    fields["deriv(Shift)"],
                    fields["SpatialMetric"],
                    fields["deriv(SpatialMetric)"],
                )
            )
            pis.append(
                pi(
                    fields["Lapse"],
                    fields["dt(Lapse)"],
                    fields["Shift"],
                    fields["dt(Shift)"],
                    fields["SpatialMetric"],
                    fields["dt(SpatialMetric)"],
                    phis[-1],
                )
            )
        self.interpolator = GhHorizonInterpolator(
            self.domain,
            0.0,
            {},
            self.element_ids,
            [self.mesh] * len(self.element_ids),
            spacetime_metrics,
            pis,
            phis,
        )

    def interpolate(self, points):
        names = [
            "InverseSpatialMetric",
            "ExtrinsicCurvature",
            "SpatialChristoffelSecondKind",
        ]
        return (
            self.interpolator(
                "unused.h5",
                "unused",
                observation=ObservationId(0),
                target_points=points,
                tensor_names=names,
                tensor_types=[
                    tnsr.II[DataVector, 3],
                    tnsr.ii[DataVector, 3],
                    tnsr.Ijj[DataVector, 3],
                ],
            ),
            names,
        )

    def test_interpolation(self):
        points = tnsr.I[DataVector, 3](
            np.array([[1.8, 0.2, 0.0], [0.1, -1.7, 0.0], [0.2, 0.3, 2.1]])
        )
        interpolated, names = self.interpolate(points)
        expected = self.solution.variables(points, names)
        for actual, name in zip(interpolated, names):
            npt.assert_allclose(actual, expected[name], atol=2.0e-5)

        # At source nodes there is no interpolation error, so the derived GH
        # fields must agree with the analytic ADM tensors to roundoff.
        points = ElementMap(self.element_ids[0], self.domain)(
            logical_coordinates(self.mesh), None, None
        )
        interpolated, names = self.interpolate(points)
        expected = self.solution.variables(points, names)
        for actual, name in zip(interpolated, names):
            npt.assert_allclose(actual, expected[name], atol=1.0e-12)

    def test_find_horizon(self):
        horizon, quantities = find_horizon(
            "unused.h5",
            "unused",
            obs_id=0,
            obs_time=0.0,
            initial_guess=Strahlkorper[Frame.Inertial](
                l_max=12, radius=2.5, center=[0.0, 0.0, 0.0]
            ),
            compute_horizon_quantities=False,
            interpolate_tensors=self.interpolator,
        )
        self.assertEqual(quantities, {})
        npt.assert_allclose(
            np.linalg.norm(cartesian_coords(horizon), axis=0), 2.0, atol=1e-3
        )

    def test_missing_points(self):
        # The helper must reject points in the excision and outside the domain.
        for x in [0.5, 3.5]:
            with self.subTest(x=x):
                with self.assertRaisesRegex(ValueError, "outside the domain"):
                    self.interpolate(
                        tnsr.I[DataVector, 3](np.array([[x], [0.0], [0.0]]))
                    )

        # A point can be inside the domain while its element data is absent.
        self.interpolator.element_data.pop(self.element_ids[0])
        with self.assertRaisesRegex(ValueError, "lack volume element data"):
            self.interpolate(
                ElementMap(self.element_ids[0], self.domain)(
                    logical_coordinates(
                        Mesh[3](1, Basis.Legendre, Quadrature.Gauss)
                    ),
                    None,
                    None,
                )
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
