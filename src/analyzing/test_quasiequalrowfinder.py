import unittest

import numpy as np

from analyzing.quasiequalrowfinder import QuasiEqualRowFinder


class TestQuasiEqualRowFinder(unittest.TestCase):

    def test1(self) -> None:
        rows = np.array([
            [1.2, 1.2, 2.1],
            [1.1, 1.1, 2.0],
            [1.0, 1.0, 1.0],
        ])
        expected_groups = [[0, 1], [2]]
        equal_row_finder = QuasiEqualRowFinder(rows, eps_list=[0.5, 0.5, 0.5])
        found_groups = sorted(sorted(g.astype(int)) for g in equal_row_finder.iter_groups())
        self.assertEqual(list(found_groups), expected_groups)
