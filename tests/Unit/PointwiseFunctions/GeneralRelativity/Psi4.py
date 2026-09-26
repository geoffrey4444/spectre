# Distributed under the MIT License.
# See LICENSE.txt for details.

import cmath
import math

import numpy as np

from .ProjectionOperators import transverse_projection_operator
from .WeylPropagating import weyl_propagating_modes


def psi_4(
    spatial_ricci,
    extrinsic_curvature,
    cov_deriv_extrinsic_curvature,
    spatial_metric,
    inv_spatial_metric,
    inertial_coords,
):
    magnitude_inertial = math.sqrt(
        np.einsum("a,b,ab", inertial_coords, inertial_coords, spatial_metric)
    )
    if magnitude_inertial != 0.0:
        r_hat = np.einsum("a", inertial_coords / magnitude_inertial)
    else:
        r_hat = np.einsum("a", inertial_coords * 0.0)

    lower_r_hat = np.einsum("a,ab", r_hat, spatial_metric)

    inv_projection_tensor = transverse_projection_operator(
        inv_spatial_metric, r_hat
    )
    projection_tensor = transverse_projection_operator(
        spatial_metric, lower_r_hat
    )
    projection_up_lo = np.einsum("ab,ac", inv_projection_tensor, spatial_metric)

    u8_plus = weyl_propagating_modes(
        spatial_ricci,
        extrinsic_curvature,
        inv_spatial_metric,
        cov_deriv_extrinsic_curvature,
        r_hat,
        inv_projection_tensor,
        projection_tensor,
        projection_up_lo,
        1,
    )

    x_coord = np.zeros((3))
    x_coord[0] = 1
    y_coord = np.zeros((3))
    y_coord[1] = 1

    def metric_cross(a, b):
        return math.sqrt(np.linalg.det(spatial_metric)) * np.linalg.solve(
            spatial_metric, np.cross(a, b)
        )

    if magnitude_inertial == 0.0:
        x_hat = x_coord / math.sqrt(spatial_metric[0, 0])
        y_hat = (
            y_coord
            - np.einsum("a,b,ab", y_coord, x_hat, spatial_metric) * x_hat
        )
        y_hat /= math.sqrt(np.einsum("a,b,ab", y_hat, y_hat, spatial_metric))
    else:
        y_hat = metric_cross(r_hat, x_coord)
        magnitude_y = math.sqrt(
            np.einsum("a,b,ab", y_hat, y_hat, spatial_metric)
        )
        minimum_magnitude = (
            100.0 * np.finfo(float).eps * math.sqrt(spatial_metric[0, 0])
        )
        use_x_direction = magnitude_y > minimum_magnitude
        if not use_x_direction:
            y_hat = metric_cross(r_hat, y_coord)
            magnitude_y = math.sqrt(
                np.einsum("a,b,ab", y_hat, y_hat, spatial_metric)
            )
        y_hat /= magnitude_y
        x_hat = metric_cross(y_hat, r_hat)
        if use_x_direction and inertial_coords[2] < 0.0:
            y_hat *= -1.0
    m_bar = x_hat - (y_hat * complex(0.0, 1.0))

    return -0.5 * np.einsum("ab,a,b", u8_plus, m_bar, m_bar)
