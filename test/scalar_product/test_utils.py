import numpy as np
import pytest

from alpsqutip.scalarprod.utils import find_linearly_independent_rows


@pytest.mark.parametrize(
    ["mat", "li_tuple"],
    (
        (
            np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0]]),
            (
                0,
                1,
            ),
        ),
        (
            np.array(
                [
                    [1.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [1.0, 0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            (
                0,
                1,
                3,
            ),
        ),
        (
            np.array(
                [
                    [1.0, 0.0, 1.0e3, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [1.0e3, 0, 1.0e6, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            (
                0,
                1,
                3,
            ),
        ),
        (
            np.array(
                [
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0],
                ]
            ),
            (0,),
        ),
    ),
)
def test_find_linearly_independent_rows(mat, li_tuple):
    assert li_tuple == find_linearly_independent_rows(mat)
