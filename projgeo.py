import math
from dataclasses import dataclass
from typing import Tuple, Iterator

from scipy.special import powm1


@dataclass
class ProjVertex:
    index: int
    x: float
    y: float
    z: float

    @property
    def xyz(self) -> Tuple[float, float, float]:
        return self.x, self.y, self.z

    def calc_dist_to_point(self, x: float, y: float) -> float:
        return math.hypot(x - self.x, y - self.y)


@dataclass
class ProjEdge:
    index: int
    p1: ProjVertex
    p2: ProjVertex

    @property
    def points(self) -> Tuple[ProjVertex, ProjVertex]:
        return self.p1, self.p2

    @property
    def z(self) -> float:
        return (self.p1.z + self.p2.z) / 2

    def calc_dist_to_point(self, x: float, y: float) -> float:
        x1, y1 = self.p1.x, self.p1.y
        x2, y2 = self.p2.x, self.p2.y

        line_mag = math.hypot(x2 - x1, y2 - y1)
        if line_mag == 0:
            return math.hypot(x - x1, y - y1)

        u = ((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)) / line_mag ** 2
        closest_x = x1 + u * (x2 - x1)
        closest_y = y1 + u * (y2 - y1)

        if u < 0.0:
            closest_x, closest_y = x1, y1
        elif u > 1.0:
            closest_x, closest_y = x2, y2

        return math.hypot(x - closest_x, y - closest_y)


@dataclass
class ProjTriangle:
    index: int
    p1: ProjVertex
    p2: ProjVertex
    p3: ProjVertex

    @property
    def points(self) -> Tuple[ProjVertex, ProjVertex, ProjVertex]:
        return self.p1, self.p2, self.p3

    @property
    def z(self) -> float:
        return (self.p1.z + self.p2.z + self.p3.z) / 3

    def contains_point(self, x: float, y: float) -> bool:
        x1, y1 = self.p1.x, self.p1.y
        x2, y2 = self.p2.x, self.p2.y
        x3, y3 = self.p3.x, self.p3.y

        p = (x, y)
        p1 = (x1, y1)
        p2 = (x2, y2)
        p3 = (x3, y3)

        d1 = self._sign(p, p1, p2)
        d2 = self._sign(p, p2, p3)
        d3 = self._sign(p, p3, p1)

        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

        return not (has_neg and has_pos)

    @staticmethod
    def _sign(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        return (x1 - x3) * (y2 - y3) - (x2 - x3) * (y1 - y3)

    def cover_vertex(self, vertex: ProjVertex) -> bool:
        x, y, z = vertex.xyz
        if not self.contains_point(x, y):
            return False  # don't cover

        if z <= self.z:
            return False  # vertex is nearer

        if vertex.index in [p.index for p in self.points]:
            return False  # vertex belongs to triangle

        return True

    def cover_edge(self, edge: ProjEdge) -> bool:
        if edge.z <= self.z:
            return False  # edge is nearer

        if {edge.p1.index, edge.p2.index} < {p.index for p in self.points}:
            return False  # edge belongs to triangle

        return True
