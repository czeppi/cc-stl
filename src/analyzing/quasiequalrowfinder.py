from typing import List, Iterator

import numpy as np

MatrixArray = np.array  # 2-dimensional array of floats
RowArray = np.array  # 1-dimensional array of floats
IndexArray = np.array  # 1-dimensional array of integers


class QuasiEqualRowFinder:
    """ A class to find quasi equal rows in a matrix

        STATUS: NOT TESTED
    """

    def __init__(self, array: MatrixArray, eps_list: List[float]):
        assert len(array.shape) == 2
        n, m = array.shape
        assert m == len(eps_list) + 1  # last column must be key, which will return

        self._array = array
        self._eps_list = eps_list
        self._num_columns = len(eps_list)

    def iter_groups(self) -> Iterator[IndexArray]:
        yield from self._iter_groups(self._array, j=0)

    def _iter_groups(self, array: MatrixArray, j: int) -> Iterator[IndexArray]:
        """
            eps = 0.5
            1.0, 1.1, 2.0, 3.0  sorted_array
            0.1, 0.9, 1.0       diff_array
            1, 2                gap_indices
            [0:2], [2:3], [3:4] group intervals
        """
        eps = self._eps_list[j]
        index_column = self._num_columns

        sorted_array = array[array[:, j].argsort()]
        diff_array = np.diff(sorted_array[:, j])
        gap_indices = np.where(diff_array >= eps)[0]
        gap_indices = np.append(gap_indices, len(sorted_array))

        k1 = 0
        for d in gap_indices:
            k2 = d + 1
            sub_array = sorted_array[k1:k2]
            if len(sub_array) >= 2:  # skip unique items
                if j + 1 == self._num_columns:
                    yield sub_array[:, index_column]
                else:
                    yield from self._iter_groups(sub_array, j=j+1)

            k1 = k2
