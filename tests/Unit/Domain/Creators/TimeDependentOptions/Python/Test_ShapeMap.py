# Distributed under the MIT License.
# See LICENSE.txt for details.

import unittest

from spectre.Domain.Creators.TimeDependentOptions import (
    KerrSchildFromBoyerLindquist,
    ShapeMapOptions,
)


class TestShapeMap(unittest.TestCase):
    def test_construction(self):
        shape_map_A = ShapeMapOptions["A"](
            10, KerrSchildFromBoyerLindquist(1.0, [0.0, 0.0, 0.99])
        )
        shape_map_B = ShapeMapOptions["B"](
            10, KerrSchildFromBoyerLindquist(1.0, [0.0, 0.0, 0.99])
        )
        shape_map_C = ShapeMapOptions["C"](
            10, KerrSchildFromBoyerLindquist(1.0, [0.0, 0.0, 0.99])
        )
        shape_map_no_object = ShapeMapOptions["None"](
            10, KerrSchildFromBoyerLindquist(1.0, [0.0, 0.0, 0.99])
        )
        self.assertNotEqual(shape_map_A, None)
        self.assertNotEqual(shape_map_B, None)
        self.assertNotEqual(shape_map_C, None)
        self.assertNotEqual(shape_map_no_object, None)

    def test_size_value_and_derivatives(self):
        shape_map_A = ShapeMapOptions["A"](
            10,
            KerrSchildFromBoyerLindquist(1.0, [0.0, 0.0, 0.99]),
            None,
            [1.2, 2.3],
        )
        shape_map_B = ShapeMapOptions["B"](
            10,
            KerrSchildFromBoyerLindquist(1.0, [0.0, 0.0, 0.99]),
            0.5,
            [1.0, 2.4],
        )
        self.assertNotEqual(shape_map_A, None)
        self.assertNotEqual(shape_map_B, None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
