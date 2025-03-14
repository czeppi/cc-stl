from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Set

from analyzing.stlmesh import StlVertex, StlEdge
from geo3d import Plane


@dataclass
class LocalAnalyzeResultData:
    planar_path: Optional[PlanarPath] = None


@dataclass
class PlanarPath:
    plane: Plane
    vertices: List[StlVertex]
    edges: List[StlEdge]

    def check_consistency(self) -> None:
        assert len(self.vertices) == len(self.edges) + 1

