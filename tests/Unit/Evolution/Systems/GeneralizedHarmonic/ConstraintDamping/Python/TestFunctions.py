# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def centered_coordinates(coords, center):
    return coords - center


def squared_distance_from_center(centered_coords, center):
    return np.einsum("i,i", centered_coords, centered_coords)


def gaussian_plus_constant_call_operator(coords, time_dependent_scale,
                                         constant, amplitude, width, center):
    one_over_width = 1.0 / width
    distance = squared_distance_from_center(
        centered_coordinates(coords, center), center)
    return amplitude * np.exp(
        -1.0 * distance * np.square(one_over_width)) + constant


def triple_gaussian_plus_constant_call_operator(coords, time_dependent_scale,
                                                constant, amplitudes, widths,
                                                center0, center1, center2):
    # Note that centers are passed in separately, because
    # check_with_random_values can only pass 1D arrays to python functions
    centers = [center0, center1, center2]
    one_over_widths = [1.0 / width for width in widths]
    distances = [
        squared_distance_from_center(centered_coordinates(coords, center),
                                     center) for center in centers
    ]
    gaussians = [
        amplitude *
        np.exp(-1.0 * distances[i] *
               np.square(one_over_widths[i] * time_dependent_scale))
        for i, amplitude in enumerate(amplitudes)
    ]
    return sum(gaussians) + constant
