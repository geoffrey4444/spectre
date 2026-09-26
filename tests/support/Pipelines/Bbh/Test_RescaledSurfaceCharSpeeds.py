# Distributed under the MIT License.
# See LICENSE.txt for details.

import os
import shutil
import unittest

import h5py
import numpy as np
import numpy.testing as npt

import spectre.IO.H5 as spectre_h5
from spectre.ApparentHorizonFinder import FastFlow, FlowType
from spectre.Domain import ElementId, ElementMap, serialize_domain
from spectre.Domain.Creators import Sphere
from spectre.Informer import unit_test_build_path
from spectre.IO.H5.IterElements import Element
from spectre.Pipelines.Bbh.RescaledSurfaceCharSpeeds import (
    rescaled_surface_char_speeds,
)
from spectre.PointwiseFunctions.AnalyticSolutions.GeneralRelativity import (
    KerrSchild,
)
from spectre.PointwiseFunctions.GeneralRelativity import spacetime_metric
from spectre.PointwiseFunctions.GeneralRelativity.GeneralizedHarmonic import (
    phi,
    pi,
)
from spectre.Spectral import Basis, Mesh, Quadrature
from spectre.Strahlkorper import (
    Frame,
    Strahlkorper,
    change_expansion_center_of_strahlkorper,
    ylm_legend_and_data,
)


def _reduce_volume_precision(filename):
    """Use the Float output type of standard evolution volume observations."""
    with h5py.File(filename, "a") as h5file:
        components = []

        def collect_components(name, obj):
            if isinstance(obj, h5py.Dataset) and name.rsplit("/", 1)[
                -1
            ].startswith(
                (
                    "SpacetimeMetric_",
                    "Pi_",
                    "Phi_",
                    "InverseSpatialMetric_",
                    "ExtrinsicCurvature_",
                    "SpatialChristoffelSecondKind_",
                )
            ):
                components.append(name)

        h5file.visititems(collect_components)
        for name in components:
            data = np.asarray(h5file[name], dtype=np.float32)
            del h5file[name]
            h5file.create_dataset(name, data=data)


class TestRescaledSurfaceCharSpeeds(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.join(
            unit_test_build_path(), "Pipelines/Bbh/RescaledSurfaceCharSpeeds"
        )
        shutil.rmtree(self.test_dir, ignore_errors=True)
        os.makedirs(self.test_dir)
        self.volume_file = os.path.join(self.test_dir, "Volume.h5")
        self.horizon_file = os.path.join(self.test_dir, "Horizons.h5")
        self.output_file = os.path.join(self.test_dir, "Diagnostic.h5")
        self.times = [0.0, 0.7, 2.0]
        self.horizons = {
            time: Strahlkorper[Frame.Inertial](
                l_max=6 + i, radius=2.0 + 0.05 * time**2, center=[0.0] * 3
            )
            for i, time in enumerate(self.times)
        }
        domain = Sphere(
            inner_radius=1.0,
            outer_radius=4.0,
            excise=True,
            initial_refinement=0,
            initial_number_of_grid_points=12,
            use_equiangular_map=True,
        ).create_domain()
        solution = KerrSchild(mass=1.0, dimensionless_spin=[0.0] * 3)
        self.domain = domain
        self.solution = solution
        self.elements = []
        elements = []
        for block_id in range(6):
            element_id = ElementId[3](block_id)
            element = Element(
                element_id,
                Mesh[3](12, Basis.Legendre, Quadrature.GaussLobatto),
                ElementMap(element_id, domain),
            )
            self.elements.append(element)
            fields = solution.variables(
                element.inertial_coordinates,
                ["Lapse", "Shift", "SpatialMetric"],
            )
            metric = spacetime_metric(
                fields["Lapse"], fields["Shift"], fields["SpatialMetric"]
            )
            elements.append(
                spectre_h5.ElementVolumeData(
                    element.id,
                    [
                        spectre_h5.TensorComponent(
                            "SpacetimeMetric" + metric.component_suffix(i),
                            metric[i],
                        )
                        for i in range(len(metric))
                    ],
                    element.mesh,
                )
            )
        with spectre_h5.H5File(self.volume_file, "w") as h5file:
            volume = h5file.insert_vol("VolumeData", version=0)
            for obs_id, time in enumerate(self.times):
                volume.write_volume_data(
                    observation_id=obs_id,
                    observation_value=time,
                    elements=elements,
                    serialized_domain=serialize_domain(domain),
                )
        with spectre_h5.H5File(self.horizon_file, "w") as h5file:
            for time, horizon in self.horizons.items():
                legend, row = ylm_legend_and_data(horizon, time, 8)
                data = h5file.try_insert_dat("AhA", legend, 0)
                data.append(row)
                h5file.close_current_object()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_saved_horizons(self):
        legend, data = rescaled_surface_char_speeds(
            self.volume_file,
            "VolumeData",
            excision_sphere="ExcisionSphere",
            horizons=self.horizons,
            number_of_surfaces=3,
            relative_excision_margin=0.05,
            output_file=self.output_file,
        )
        self.assertEqual(
            legend,
            ["Time", "Status"]
            + [
                name + "_" + str(i)
                for i in range(3)
                for name in ["RadiusFactor", "MinCharSpeed", "MaxCharSpeed"]
            ],
        )
        npt.assert_allclose(data[:, 0], self.times)
        npt.assert_array_equal(data[:, 1], [1, 0, 0])
        self.assertTrue(np.isnan(data[0, [3, 4, 6, 7, 9, 10]]).all())
        # The third sample differentiates an exact quadratic on uneven times,
        # despite the changing angular resolution. The second uses two points.
        for row, radius, velocity in zip(data[1:], [2.0245, 2.2], [0.035, 0.2]):
            factors = 1.0 - (np.arange(3) / 2.0) ** 2 * (1.0 - 1.05 / radius)
            npt.assert_allclose(row[2::3], factors)
            radii = factors * radius
            expected = (2.0 / radii - 1.0) / np.sqrt(1.0 + 2.0 / radii)
            expected += np.sqrt(1.0 + 2.0 / radii) * factors * velocity
            npt.assert_allclose(row[3::3], expected, atol=2e-4)
            npt.assert_allclose(row[4::3], expected, atol=2e-4)
        with spectre_h5.H5File(self.output_file, "r") as h5file:
            output = h5file.get_dat("RescaledCharSpeeds")
            self.assertEqual(output.get_legend(), legend)
            npt.assert_allclose(output.get_data(), data)
        saved_legend, saved_data = rescaled_surface_char_speeds(
            self.volume_file,
            "VolumeData",
            excision_sphere="ExcisionSphere",
            horizon_file=self.horizon_file,
            horizon_subfile="AhA",
            number_of_surfaces=3,
            relative_excision_margin=0.05,
        )
        self.assertEqual(saved_legend, legend)
        npt.assert_allclose(saved_data, data)
        # Selecting only one volume observation still uses earlier saved
        # horizon history. Do not turn this into a spurious startup row.
        _, selected = rescaled_surface_char_speeds(
            self.volume_file,
            "VolumeData",
            excision_sphere="ExcisionSphere",
            horizons=self.horizons,
            observation_ids=[2],
            number_of_surfaces=3,
            relative_excision_margin=0.05,
        )
        npt.assert_allclose(selected, data[-1:])
        # Different expansion centers of saved surfaces are representation
        # choices. Recenter before differentiating to recover the same motion.
        recentered_horizons = {
            time: change_expansion_center_of_strahlkorper(
                surface, [0.02 * time, 0.0, 0.0]
            )
            for time, surface in self.horizons.items()
        }
        _, recentered_data = rescaled_surface_char_speeds(
            self.volume_file,
            "VolumeData",
            excision_sphere="ExcisionSphere",
            horizons=recentered_horizons,
            observation_ids=[2],
            number_of_surfaces=3,
            relative_excision_margin=0.05,
        )
        npt.assert_allclose(recentered_data, selected, atol=1e-8)

    def test_distorted_horizons_and_explicit_derivative(self):
        horizons = {
            time: Strahlkorper[Frame.Distorted](
                l_max=6, radius=2.0, center=[0.0] * 3
            )
            for time in self.times
        }
        derivative = Strahlkorper[Frame.Distorted](
            l_max=6, radius=0.0, center=[0.0] * 3
        )
        _, data = rescaled_surface_char_speeds(
            self.volume_file,
            "VolumeData",
            "ExcisionSphere",
            horizons=horizons,
            time_derivatives={0.0: derivative},
            number_of_surfaces=2,
            relative_excision_margin=0.05,
        )
        npt.assert_array_equal(data[:, 1], 0)
        npt.assert_allclose(data[:, 3:5], 0.0, atol=2e-4)
        self.assertTrue((data[:, 6:] > 0.0).all())
        with spectre_h5.H5File(self.horizon_file, "a") as h5file:
            for time, surface in horizons.items():
                legend, row = ylm_legend_and_data(surface, time, 6)
                output = h5file.try_insert_dat("DistortedAhA", legend, 0)
                output.append(row)
                h5file.close_current_object()
        _, saved_data = rescaled_surface_char_speeds(
            self.volume_file,
            "VolumeData",
            "ExcisionSphere",
            horizon_file=self.horizon_file,
            horizon_subfile="DistortedAhA",
            horizon_frame=Frame.Distorted,
            time_derivatives={0.0: derivative},
            number_of_surfaces=2,
            relative_excision_margin=0.05,
        )
        npt.assert_allclose(saved_data, data)

    def test_float_volume_data(self):
        _, double_data = rescaled_surface_char_speeds(
            self.volume_file,
            "VolumeData",
            "ExcisionSphere",
            horizons=self.horizons,
            observation_ids=[2],
        )
        _reduce_volume_precision(self.volume_file)
        with spectre_h5.H5File(self.volume_file, "r") as h5file:
            component = h5file.get_vol("VolumeData").get_tensor_component(
                2, "SpacetimeMetric_tt"
            )
            self.assertEqual(component.data.dtype, np.dtype("float32"))
        _, float_data = rescaled_surface_char_speeds(
            self.volume_file,
            "VolumeData",
            "ExcisionSphere",
            horizons=self.horizons,
            observation_ids=[2],
        )
        npt.assert_allclose(float_data, double_data, atol=1e-6)

    def test_validation_and_statuses(self):
        arguments = dict(
            h5_files=self.volume_file,
            subfile_name="VolumeData",
            excision_sphere="ExcisionSphere",
            horizons=self.horizons,
            observation_ids=[2],
        )
        for override, message in [
            ({"number_of_surfaces": 1}, "At least two"),
            ({"relative_excision_margin": 0.0}, "must be positive"),
            ({"time_tolerance": -1.0}, "must be finite and nonnegative"),
            ({"observation_ids": [5]}, "Observation IDs not found"),
            ({"observation_ids": [2, 2]}, "must not contain duplicates"),
            ({"horizons": {0.0: self.horizons[0.0]}}, "Expected one horizon"),
            ({"horizon_frame": Frame.Grid}, "Inertial or Distorted"),
        ]:
            with self.subTest(override=override):
                with self.assertRaisesRegex(ValueError, message):
                    rescaled_surface_char_speeds(**(arguments | override))
        with self.assertRaisesRegex(ValueError, "DoesNotExist"):
            rescaled_surface_char_speeds(
                **(arguments | {"spacetime_metric_name": "DoesNotExist"})
            )
        _, missing_coverage = rescaled_surface_char_speeds(
            **arguments, blocks=[]
        )
        self.assertEqual(missing_coverage[0, 1], 4)
        self.assertTrue(np.isnan(missing_coverage[0, 3::3]).all())
        for radius, expected_status in [(0.8, 2), (5.0, 3)]:
            horizons = {
                time: Strahlkorper[Frame.Distorted](
                    l_max=6, radius=radius, center=[0.0] * 3
                )
                for time in self.times
            }
            _, data = rescaled_surface_char_speeds(
                **(arguments | {"horizons": horizons})
            )
            self.assertEqual(data[0, 1], expected_status)

    def test_refind_horizon(self):
        # Add the fields needed by both ways to re-find the horizon. Keep the
        # other tests on metric-only data to enforce their weaker requirements.
        elements = []
        for element in self.elements:
            fields = self.solution.variables(
                element.inertial_coordinates,
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
                    "InverseSpatialMetric",
                    "ExtrinsicCurvature",
                    "SpatialChristoffelSecondKind",
                ],
            )
            tensors = {
                name: fields[name]
                for name in [
                    "InverseSpatialMetric",
                    "ExtrinsicCurvature",
                    "SpatialChristoffelSecondKind",
                ]
            }
            tensors["SpacetimeMetric"] = spacetime_metric(
                fields["Lapse"], fields["Shift"], fields["SpatialMetric"]
            )
            tensors["Phi"] = phi(
                fields["Lapse"],
                fields["deriv(Lapse)"],
                fields["Shift"],
                fields["deriv(Shift)"],
                fields["SpatialMetric"],
                fields["deriv(SpatialMetric)"],
            )
            tensors["Pi"] = pi(
                fields["Lapse"],
                fields["dt(Lapse)"],
                fields["Shift"],
                fields["dt(Shift)"],
                fields["SpatialMetric"],
                fields["dt(SpatialMetric)"],
                tensors["Phi"],
            )
            elements.append(
                spectre_h5.ElementVolumeData(
                    element.id,
                    [
                        spectre_h5.TensorComponent(
                            name + tensor.component_suffix(i), tensor[i]
                        )
                        for name, tensor in tensors.items()
                        for i in range(len(tensor))
                    ],
                    element.mesh,
                )
            )
        filename = os.path.join(self.test_dir, "HorizonFields.h5")
        with spectre_h5.H5File(filename, "w") as h5file:
            volume = h5file.insert_vol("VolumeData", version=0)
            for obs_id, time in enumerate(self.times[:2]):
                volume.write_volume_data(
                    observation_id=obs_id,
                    observation_value=time,
                    elements=elements,
                    serialized_domain=serialize_domain(self.domain),
                )
        _, supplied = rescaled_surface_char_speeds(
            filename,
            "VolumeData",
            "ExcisionSphere",
            horizons={
                time: Strahlkorper[Frame.Inertial](
                    l_max=8, radius=2.0, center=[0.0] * 3
                )
                for time in self.times[:2]
            },
            number_of_surfaces=3,
            relative_excision_margin=0.05,
        )
        _reduce_volume_precision(filename)
        for horizon_data in ["Adm", "Gh"]:
            with self.subTest(horizon_data=horizon_data):
                _, data = rescaled_surface_char_speeds(
                    filename,
                    "VolumeData",
                    "ExcisionSphere",
                    initial_guess=Strahlkorper[Frame.Inertial](
                        l_max=8, radius=2.3, center=[0.0] * 3
                    ),
                    fast_flow=FastFlow(
                        FlowType.Fast,
                        alpha=1.0,
                        beta=0.5,
                        abs_tol=1e-6,
                        truncation_tol=0.01,
                        divergence_tol=1.2,
                        divergence_iter=5,
                        max_its=100,
                    ),
                    horizon_data=horizon_data,
                    number_of_surfaces=3,
                    relative_excision_margin=0.05,
                )
                npt.assert_allclose(data, supplied, atol=1e-3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
