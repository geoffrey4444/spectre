# Distributed under the MIT License.
# See LICENSE.txt for details.

import unittest

from spectre.Domain.Creators.TimeDependentOptions import (
    KerrSchildFromBoyerLindquist,
    KerrSchildFromHorizon,
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

        for label in ["A", "B", "C", "None"]:
            shape_map = ShapeMapOptions[label](
                28, KerrSchildFromHorizon(1.0, [0.0, 0.0, 0.99])
            )
            self.assertIsNotNone(shape_map)


if __name__ == "__main__":
    unittest.main(verbosity=2)
