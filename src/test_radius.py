from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, Iterator, Optional


EPS = 1e-6


@dataclass
class Point2D:
    x: float
    y: float

    def __iter__(self) -> Iterator[float]:
        yield self.x
        yield self.y

    def __add__(self, other: Point2D) -> Point2D:
        return Point2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Point2D) -> Point2D:
        return Point2D(self.x - other.x, self.y - other.y)

    def __mul__(self, s: float) -> Point2D:
        return Point2D(s * self.x, s * self.y)

    def __rmul__(self, s: float) -> Point2D:
        return Point2D(s * self.x, s * self.y)

    def __truediv__(self, s: float) -> Point2D:
        return Point2D(self.x / s, self.y / s)

    @property
    def length(self) -> float:
        return math.hypot(self.x, self.y)


def main():
    for p1, p2, p3 in iter_triples():
        calc_circle(p1, p2, p3)


def iter_triples() -> Iterator[Point2D]:
    points = list(iter_points())
    for i in range(len(points) - 2):
        yield points[i], points[i + 1], points[i + 2]


def iter_points() -> Iterator[Point2D]:
    x = 0.0
    y = 0.0
    for dx, dy in iter_coordinates():
        x += dx
        y += dy
        yield Point2D(x, y)


def iter_coordinates() -> Iterator[Tuple[int, int]]:
    """ some points of KLM stem """
    yield 0, 0
    yield 7, 51
    yield 20, 48
    yield 32, 41
    yield 41, 31
    yield 48, 20
    yield 51, 7


def calc_circle(p1: Point2D, p2: Point2D, p3: Point2D) -> None:
    dx21 = p2.x - p1.x
    dy21 = p2.y - p1.y
    dx32 = p3.x - p2.x
    dy32 = p3.y - p2.y

    p12 = (p1 + p2) / 2
    p23 = (p2 + p3) / 2
    p12b = p12 + Point2D(dy21, -dx21)
    p23b = p23 + Point2D(dy32, -dx32)
    center = calc_line_intersections(p12, p12b, p23, p23b)
    r1 = (p1 - center).length
    r2 = (p2 - center).length
    r3 = (p3 - center).length
    assert abs(r2 - r1) < EPS
    assert abs(r3 - r2) < EPS

    print(f'center: ({center.x:.3f}, {center.y:.3f}), radius: {r1:.3f}')


def calc_line_intersections(p1: Point2D, p2: Point2D, p3: Point2D, p4: Point2D) -> Optional[Point2D]:
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    dx21 = x2 - x1
    dx43 = x4 - x3
    dy21 = y2 - y1
    dy43 = y4 - y3
    denom = dx21 * dy43 - dy21 * dx43
    if abs(denom) < EPS:
        return

    xy12 = x1 * y2 - y1 * x2
    xy34 = x3 * y4 - y3 * x4

    xs = (dx21 * xy34 - dx43 * xy12) / denom
    ys = (dy21 * xy34 - dy43 * xy12) / denom
    return Point2D(xs, ys)

    
main()
