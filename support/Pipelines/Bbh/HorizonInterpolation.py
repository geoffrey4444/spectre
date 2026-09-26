# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np

from spectre.DataStructures.Tensor.EagerMath import determinant_and_inverse
from spectre.Domain import (
    block_logical_coordinates,
    element_logical_coordinates,
)
from spectre.Interpolation import Irregular
from spectre.PointwiseFunctions.GeneralRelativity import (
    lapse,
    shift,
    spacetime_normal_vector,
    spatial_metric,
)
from spectre.PointwiseFunctions.GeneralRelativity.GeneralizedHarmonic import (
    christoffel_second_kind,
    extrinsic_curvature,
)


class GhHorizonInterpolator:
    """Interpolate horizon-finding fields derived from nodal GH variables.

    The arguments 'element_ids', 'meshes', 'spacetime_metrics', 'pis', and
    'phis' are matching lists of elements and their nodal fields at 'time'.
    The GH tensors must be in the Inertial frame. Fields needed by the horizon
    finder are computed at volume nodes once, before interpolation. The
    domain and functions of time locate Inertial target points in the elements.

    Instances have the signature of
    'spectre.IO.Exporter.interpolate_tensors_to_points' for use with
    'find_horizon(..., compute_horizon_quantities=False,
    interpolate_tensors=...)'. The file and observation arguments are unused;
    each instance holds data for a single observation. Only the default
    inverse-spatial-metric, extrinsic-curvature, and spatial-Christoffel names
    are supported.
    """

    def __init__(
        self,
        domain,
        time,
        functions_of_time,
        element_ids,
        meshes,
        spacetime_metrics,
        pis,
        phis,
    ):
        self.domain = domain
        self.time = time
        self.functions_of_time = functions_of_time
        if not element_ids or any(
            len(data) != len(element_ids)
            for data in [meshes, spacetime_metrics, pis, phis]
        ):
            raise ValueError("Supply matching, nonempty lists of element data.")
        if len(set(element_ids)) != len(element_ids):
            raise ValueError("Element IDs must be unique.")
        self.element_data = {}
        for element_id, mesh, metric, pi, phi in zip(
            element_ids, meshes, spacetime_metrics, pis, phis
        ):
            if any(
                len(component) != mesh.number_of_grid_points()
                for tensor in [metric, pi, phi]
                for component in tensor
            ):
                raise ValueError("Tensor sizes must match their element mesh.")
            _, inv_spatial_metric = determinant_and_inverse(
                spatial_metric(metric)
            )
            inertial_shift = shift(metric, inv_spatial_metric)
            normal = spacetime_normal_vector(
                lapse(inertial_shift, metric), inertial_shift
            )
            self.element_data[element_id] = (
                mesh,
                {
                    "InverseSpatialMetric": inv_spatial_metric,
                    "ExtrinsicCurvature": extrinsic_curvature(normal, pi, phi),
                    "SpatialChristoffelSecondKind": christoffel_second_kind(
                        phi, inv_spatial_metric
                    ),
                },
            )

    def __call__(
        self,
        h5_files,
        subfile_name,
        *,
        observation,
        target_points,
        tensor_names,
        tensor_types,
    ):
        if len(tensor_names) != len(tensor_types):
            raise ValueError("Supply a tensor type for every tensor name.")
        supported_names = next(iter(self.element_data.values()))[1]
        for name in tensor_names:
            if name not in supported_names:
                raise ValueError(f"Unsupported horizon-finding tensor: {name}")
        block_logical_coords = block_logical_coordinates(
            self.domain,
            target_points,
            time=self.time,
            functions_of_time=self.functions_of_time,
        )
        if any(coords is None for coords in block_logical_coords):
            raise ValueError("Horizon target points are outside the domain.")
        logical_coords = element_logical_coordinates(
            list(self.element_data), block_logical_coords
        )
        num_points = len(target_points[0])
        covered = np.zeros(num_points, dtype=bool)
        result = [
            np.empty((tensor_type().size, num_points))
            for tensor_type in tensor_types
        ]
        for element_id, coords in logical_coords.items():
            mesh, fields = self.element_data[element_id]
            interpolant = Irregular[3](
                mesh, list(coords.element_logical_coords)
            )
            covered[coords.offsets] = True
            for tensor_data, name in zip(result, tensor_names):
                for i, component in enumerate(fields[name]):
                    tensor_data[i, coords.offsets] = interpolant.interpolate(
                        component
                    )
        if not np.all(covered):
            raise ValueError("Horizon target points lack volume element data.")
        return [
            tensor_type(data) for tensor_type, data in zip(tensor_types, result)
        ]
