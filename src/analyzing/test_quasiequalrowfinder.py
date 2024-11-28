import unittest

import numpy as np

from analyzing.quasiequalrowfinder import QuasiEqualRowFinder


class TestQuasiEqualRowFinder(unittest.TestCase):

    def test1(self) -> None:
        rows = np.array([
            [1.2, 1.2, 2.1, 7],
            [1.1, 1.1, 2.0, 9],
            [1.0, 1.0, 1.0, 8],
        ])
        expected_groups = [[7, 9]]
        equal_row_finder = QuasiEqualRowFinder(rows, eps_list=[0.5, 0.5, 0.5])
        found_groups = sorted(sorted(g.astype(int)) for g in equal_row_finder.iter_groups())
        self.assertEqual(list(found_groups), expected_groups)

    def test2(self) -> None:
        rows = np.array([
            [1.0, 1.0, 1.0, 11],
            [1.1, 1.0, 1.1, 12],
            [1.0, 1.1, 2.0, 21],
            [1.1, 1.0, 2.1, 22],
            [1.2, 2.0, 1.0, 31],
            [1.1, 2.1, 1.1, 32],
            [2.0, 1.1, 2.1, 41],
            [2.1, 1.0, 2.0, 42],
        ])
        expected_groups = [[11, 12], [21, 22], [31, 32], [41, 42]]
        equal_row_finder = QuasiEqualRowFinder(rows, eps_list=[0.5, 0.5, 0.5])
        found_groups = sorted(sorted(g.astype(int)) for g in equal_row_finder.iter_groups())
        self.assertEqual(list(found_groups), expected_groups)
