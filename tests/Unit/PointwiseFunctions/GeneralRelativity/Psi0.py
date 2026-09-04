# Distributed under the MIT License.
# See LICENSE.txt for details.

import math

import numpy as np

from .ProjectionOperators import transverse_projection_operator
from .WeylPropagating import weyl_propagating_modes


def psi_0_real(
    spatial_ricci,
    extrinsic_curvature,
    cov_deriv_extrinsic_curvature,
    spatial_metric,
    inverse_spatial_metric,
    inertial_coords,
):
    magnitude_inertial = math.sqrt(
        np.einsum("a,b,ab", inertial_coords, inertial_coords, spatial_metric)
    )
    if magnitude_inertial != 0.0:
        r_hat = inertial_coords / magnitude_inertial
    else:
        r_hat = inertial_coords * 0.0

    lower_r_hat = np.einsum("a,ab", r_hat, spatial_metric)
    inverse_projection_tensor = transverse_projection_operator(
        inverse_spatial_metric, r_hat
    )
    projection_tensor = transverse_projection_operator(
        spatial_metric, lower_r_hat
    )
    projection_up_lo = np.einsum(
        "ab,ac", inverse_projection_tensor, spatial_metric
    )
    u8_minus = weyl_propagating_modes(
        spatial_ricci,
        extrinsic_curvature,
        inverse_spatial_metric,
        cov_deriv_extrinsic_curvature,
        r_hat,
        inverse_projection_tensor,
        projection_tensor,
        projection_up_lo,
        -1,
    )

    x_coord = np.zeros(3)
    x_coord[0] = 1.0
    x_component = np.einsum("a,b,ab", x_coord, r_hat, spatial_metric)
    x_hat = x_coord - x_component * r_hat
    magnitude_x = math.sqrt(np.einsum("a,b,ab", x_hat, x_hat, spatial_metric))
    minimum_magnitude = (
        100.0 * np.finfo(float).eps * math.sqrt(spatial_metric[0, 0])
    )

    y_coord = np.zeros(3)
    y_coord[1] = 1.0
    y_component = np.einsum("a,b,ab", y_coord, r_hat, spatial_metric)
    projected_y = y_coord - y_component * r_hat
    if magnitude_x > minimum_magnitude:
        x_hat /= magnitude_x
    else:
        x_hat = projected_y
        magnitude_x = math.sqrt(
            np.einsum("a,b,ab", x_hat, x_hat, spatial_metric)
        )
        if magnitude_x != 0.0:
            x_hat /= magnitude_x
        else:
            x_hat *= 0.0
    y_component = np.einsum("a,b,ab", projected_y, x_hat, spatial_metric)
    y_hat = projected_y - y_component * x_hat
    magnitude_y = math.sqrt(np.einsum("a,b,ab", y_hat, y_hat, spatial_metric))
    minimum_magnitude_y = (
        100.0 * np.finfo(float).eps * math.sqrt(spatial_metric[1, 1])
    )
    if magnitude_y > minimum_magnitude_y:
        y_hat /= magnitude_y
    elif magnitude_inertial != 0.0:
        y_hat = math.sqrt(np.linalg.det(spatial_metric)) * np.einsum(
            "li,l->i", inverse_spatial_metric, np.cross(r_hat, x_hat)
        )
    else:
        y_hat *= 0.0

    return -0.5 * np.einsum(
        "ab,ab", u8_minus, np.outer(x_hat, x_hat) - np.outer(y_hat, y_hat)
    )
