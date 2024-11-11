import unittest

from projgeo import ProjTriangle, ProjVertex


class TestProjTriangleContainsPoint(unittest.TestCase):

    def test1(self) -> None:
        triangle = self._create_triangle()
        self.assertTrue(triangle.contains_point(1, 1))

    def test2(self) -> None:
        triangle = self._create_triangle()
        self.assertFalse(triangle.contains_point(3, 3))

    @staticmethod
    def _create_triangle() -> ProjTriangle:
        return ProjTriangle(index=0,
                            p1=ProjVertex(index=0, x=0, y=0, z=0),
                            p2=ProjVertex(index=0, x=4, y=0, z=0),
                            p3=ProjVertex(index=0, x=0, y=4, z=0))