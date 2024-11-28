from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Iterator, Tuple, Any, Callable

import trimesh
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QToolBar, QWidget

from analyzing.analyzeresult import AnalyzeResultData, AnalyzeResult
from analyzing.geoanalyzer import GeoAnalyzer
from camera import Camera
from geo3d import Plane, Sphere
from itemdetectoratmousepos import MeshItemKey, MeshItemType
from meshcolorizer import MeshColorizer


@dataclass
class MeshInfoWinHandlers:
    change_colorizer: Callable[[MeshColorizer], None]


class MeshInfoWin(QWidget):
    """
    No cur item:
      - Button "show recognized faces"
      - Button "show recognized polylines"

    cur item is face:
      - show index, norm angles + dist. to source
      if plane:
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
        self._handlers: Optional[MeshInfoWinHandlers] = None

        self._camera: Optional[Camera] = None
        self._cur_item: Optional[MeshItemKey] = None
        self._sel_items: List[MeshItemKey] = []
        self._analyze_result: Optional[AnalyzeResultData] = None

        self._toolbar = self._create_toolbar()
        self._label = QLabel()
        self._update_label()

        layout = self._create_layout()
        self.setLayout(layout)

    def _create_toolbar(self) -> QToolBar:
        toolbar = QToolBar(parent=self)  # important for vlayout.setMenuBar(toolbar)
        analyze_button = self._create_analyze_button()
        self._analyze_action = toolbar.addWidget(analyze_button)
        # for widget in self._iter_toolbar_widgets():
        #     toolbar.addWidget(widget)
        return toolbar

    def _iter_toolbar_widgets(self) -> Iterator[QWidget]:
        yield self._create_analyze_button()

    def _create_analyze_button(self) -> QPushButton:
        button = QPushButton()
        button.setText('analyze')
        #button.setStyleSheet('color: red; background-color: lightgray')
        #button.setToolTip('switch solve parameter to "expected", to edit data')
        button.clicked.connect(self.on_analyze)
        return button

    def _create_layout(self):
        margin = 5
        vlayout = QVBoxLayout()
        vlayout.setContentsMargins(margin, 0, margin, margin)
        vlayout.setMenuBar(self._toolbar)
        vlayout.addWidget(self._label)
        return vlayout

    def set_handlers(self, handlers: MeshInfoWinHandlers) -> None:
        self._handlers = handlers

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
                                           cur_item=self._cur_item, sel_items=self._sel_items,
                                           analyze_result=self._analyze_result)
        html_text = html_creator.create_html()
        self._label.setText(html_text)
        self._label.setAlignment(Qt.AlignTop | Qt.AlignLeft)

    def on_analyze(self) -> None:
        geo_analyzer = GeoAnalyzer(self._mesh)
        result_data = geo_analyzer.analyze()
        self._analyze_result = AnalyzeResult(result_data)
        self._toolbar.removeAction(self._analyze_action)
        self._update_label()
        if self._handlers.change_colorizer:
            colorizer = MeshColorizer(mesh=self._mesh, analyze_result=self._analyze_result)
            self._handlers.change_colorizer(colorizer)


class MeshInfoHtmlCreator:

    def __init__(self, mesh: trimesh.Trimesh, camera: Camera,
                 cur_item: Optional[MeshItemKey], sel_items: List[MeshItemKey],
                 analyze_result: Optional[AnalyzeResult]):
        self._mesh = mesh
        self._camera = camera
        self._cur_item = cur_item
        self._sel_items = sel_items
        self._analyze_result = analyze_result

    def create_html(self) -> str:
        return '\n'.join(self._iter_lines())

    def _iter_lines(self) -> Iterator[str]:
        yield from self._cur_item_infos()
        yield from self._camera_infos()
        yield from self._analyze_infos()
        yield from self._iter_mesh_statistics()

    def _cur_item_infos(self) -> Iterator[str]:
        cur_item = self._cur_item
        if cur_item is None:
            return

        yield '<h1>cur item</h1>'
        yield '<table>'

        for name, value in self._cur_item_rows(cur_item):
            yield f'<tr>'
            yield f'<td align="left">{name}: </td>'
            yield f'<td>{value}</td>'
            yield f'</tr>'

        yield '</table>'

    def _cur_item_rows(self, cur_item: MeshItemKey) -> Iterator[Tuple[str, str]]:
        if cur_item.type == MeshItemType.VERTEX:
            yield from self._vertex_rows(cur_item.index)
        elif cur_item.type == MeshItemType.EDGE:
            yield from self._edge_rows(cur_item.index)
        elif cur_item.type == MeshItemType.FACE:
            yield from self._face_rows(cur_item.index)

    def _vertex_rows(self, vertex_index: int) -> Iterator[Tuple[str, str]]:
        mesh_vertex = self._mesh.vertices[vertex_index]
        x, y, z = mesh_vertex

        yield 'vertex', str(vertex_index)
        yield f'x, y, z', f'({x:.3f}, {y:.3f}, {z:.3f})'

    def _edge_rows(self, edge_index: int) -> Iterator[Tuple[str, str]]:
        mesh_edge = self._mesh.edges_unique[edge_index]
        vertex1_index, vertex2_index = mesh_edge
        mesh_vertex1 = self._mesh.vertices[vertex1_index]
        mesh_vertex2 = self._mesh.vertices[vertex2_index]
        x1, y1, z1 = mesh_vertex1
        x2, y2, z2 = mesh_vertex2

        yield f'edge', str(edge_index)
        yield f'dx, dy, dz', f'({x2 - x1:.3f}, {y2 - y1:.3f}, {z2 - z1:.3f})'

    def _face_rows(self, face_index: int) -> Iterator[Tuple[str, str]]:
        face_normal = self._mesh.face_normals[face_index]
        nx, ny, nz = face_normal
        yield f'face', str(face_index)
        yield f'normal', f'({nx:.3f}, {ny:.3f}, {nz:.3f})'

        if self._analyze_result:
            surface_patch = self._analyze_result.find_surface_patch(face_index)
            if surface_patch and isinstance(surface_patch.form, Plane):
                plane = surface_patch.form
                nx, ny, nz = plane.normal
                yield f'plane.normal', f'({nx:.3f}, {ny:.3f}, {nz:.3f})'
                yield f'plane.distance', f'{plane.distance:.3f}'
                yield 'num triangles', f'{len(surface_patch.triangle_indices)}'
            elif surface_patch and isinstance(surface_patch.form, Sphere):
                sphere = surface_patch.form
                cx, cy, cz = sphere.center
                yield f'sphere.center', f'({cx:.3f}, {cy:.3f}, {cz:.3f})'
                yield f'sphere.radius', f'{sphere.radius:.3f}'
                yield 'num triangles', f'{len(surface_patch.triangle_indices)}'

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

    def _analyze_infos(self) -> Iterator[str]:
        result = self._analyze_result
        if result is None:
            return

        yield '<h1>analyze result</h1>'
        yield '<table>'

        for name, value in self._iter_analyze_rows():
            yield f'<tr>'
            yield f'<td align="left">{name}: </td>'
            yield f'<td>{value}</td>'
            yield f'</tr>'

        yield '</table>'

    def _iter_analyze_rows(self) -> Iterator[Tuple[str, str]]:
        result = self._analyze_result

        yield 'num planes', result.count_planes()
        yield 'num spheres', result.count_spheres()
        yield 'num cylinders', result.count_cylinders()

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

        x_min = self._mesh.vertices[:, 0].min()
        x_max = self._mesh.vertices[:, 0].max()
        y_min = self._mesh.vertices[:, 1].min()
        y_max = self._mesh.vertices[:, 1].max()
        z_min = self._mesh.vertices[:, 2].min()
        z_max = self._mesh.vertices[:, 2].max()

        yield 'num vertices', len(mesh.vertices)
        yield 'num edges', len(mesh.edges_unique)
        yield 'num triangles', len(mesh.faces)
        yield 'x-interval', f'[{x_min:.3f}, {x_max:.3f}]'
        yield 'y-interval', f'[{y_min:.3f}, {y_max:.3f}]'
        yield 'z-interval', f'[{z_min:.3f}, {z_max:.3f}]'

