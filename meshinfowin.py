from __future__ import annotations

from typing import Optional, List, Iterator, Tuple, Any

import trimesh
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel

from camera import Camera
from itemdetectoratmousepos import MeshItemKey, MeshItemType


class MeshInfoWin(QLabel):
    """
    No cur item:
      - Button "show recocnized faces"
      - Button "show recocnized polylines"

    cur item is face:
      - show index, norm angles + dist. to source
      if level/floor:
        - buttons: "show connected", "show whole", "show all ortho"
      elif sphere:
        - show center + radius
        - buttons: "show connected", "show whole sphere", "show all sphere with same radius"
      elif cylinder (tube etc.):
        - show axis + radius
        - buttons: "show whole cylinder", "show all cylinders with same radius", "show all sphere with same radius"

    cur item is edge:
      - show index, start point, end point, length, delta
      - show level (if polyline)
      - button "show polyline"
      if line:
        - buttons: "show connected", "show whole line", "show all parallels", "show all orthogonals"
      elif circle:
        - show center, radius
        - buttons: "show connected", "show whole cirle", "show all parallels", "show all same radius"
      elif bezier:
        - show parameters
        - buttons: "show connected", "show whole bezier", "show all parallels", "show congruent"

    cur item == vertex:
      - show: index, position
    """

    def __init__(self, mesh: trimesh.Trimesh):
        super().__init__()
        self._mesh = mesh
        self._camera: Optional[Camera] = None
        self._cur_item: Optional[MeshItemKey] = None
        self._sel_items: List[MeshItemKey] = []

    def on_opengl_change_camera_pos(self, camera: Camera) -> None:
        self._camera = camera
        self._update_label()

    def on_opengl_change_cur_item(self, cur_item: Optional[MeshItemKey]) -> None:
        self._cur_item = cur_item
        self._update_label()

    def on_opengl_change_sel_items(self, new_sel_items: List[MeshItemKey]) -> None:
        self._sel_items = new_sel_items
        self._update_label()

    def _update_label(self) -> None:
        html_creator = MeshInfoHtmlCreator(mesh=self._mesh, camera=self._camera,
                                           cur_item=self._cur_item, sel_items=self._sel_items)
        html_text = html_creator.create_html()
        self.setText(html_text)
        self.setAlignment(Qt.AlignTop | Qt.AlignLeft)


class MeshInfoHtmlCreator:

    def __init__(self, mesh: trimesh.Trimesh, camera: Camera,
                 cur_item: Optional[MeshItemKey], sel_items: List[MeshItemKey]):
        self._mesh = mesh
        self._camera = camera
        self._cur_item = cur_item
        self._sel_items = sel_items

    def create_html(self) -> str:
        return '\n'.join(self._iter_lines())

    def _iter_lines(self) -> Iterator[str]:
        yield from self._iter_mesh_statistics()
        yield from self._camera_infos()
        yield from self._cur_item_infos()

    def _iter_mesh_statistics(self) -> Iterator[str]:
        yield '<h1>mesh</h1>'
        yield '<table>'

        for name, value in self._iter_mesh_statistics_rows():
            yield f'<tr>'
            yield f'<td align="left">{name}: </td>'
            yield f'<td>{value}</td>'
            yield f'</tr>'

        yield '</table>'

    def _iter_mesh_statistics_rows(self) -> Iterator[Tuple[str, Any]]:
        mesh = self._mesh

        yield 'num vertices', len(mesh.vertices)
        yield 'num edges', len(mesh.edges_unique)
        yield 'num triangles', len(mesh.faces)

    def _camera_infos(self) -> Iterator[str]:
        camera = self._camera
        if camera is None:
            return

        yield '<h1>camera</h1>'
        yield '<table>'

        for name, value in self._iter_camera_rows():
            yield f'<tr>'
            yield f'<td align="left">{name}: </td>'
            yield f'<td>{value}</td>'
            yield f'</tr>'

        yield '</table>'

    def _iter_camera_rows(self) -> Iterator[Tuple[str, str]]:
        camera = self._camera

        yield 'distance', f'{camera.distance:.1f}'
        yield 'azimuth', f'{camera.azimuth:.1f}'
        yield 'elevation', f'{camera.elevation:.1f}'

    def _cur_item_infos(self) -> Iterator[str]:
        cur_item = self._cur_item
        if cur_item is None:
            return

        yield '<h1>cur item</h1>'

        if cur_item.type == MeshItemType.VERTEX:
            mesh_vertex = self._mesh.vertices[cur_item.index]
            x, y, z = mesh_vertex
            yield f'vertex[{cur_item.index}]:\n({x:.3f}, {y:.3f}, {z:.3f})'
        elif cur_item.type == MeshItemType.EDGE:
            yield f'edge[{cur_item.index}]'
        elif cur_item.type == MeshItemType.FACE:
            yield f'face[{cur_item.index}]'
