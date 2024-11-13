from __future__ import annotations

import sys
from cProfile import label
from typing import Callable, List, Optional

import trimesh
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMainWindow, QApplication, QSplitter, QLabel, QFileDialog

from itemdetectoratmousepos import MeshItemType, MeshItemKey
from openglwidget import GL_VIEW_SIZE, OpenGlWin, OpenGlWinHandlers

STL_PATH = "stl-files/KLP_Lame_Tilted.stl"
#STL_PATH = "stl-files/charybdisnano_v2_v187.stl"
#STL_PATH = "stl-files/adapter_v2_bottom_pmw_3389.stl"


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("STL Example")
        self.resize(*GL_VIEW_SIZE)

        self._mesh = self._read_mesh(STL_PATH)

        self._add_menubar()

        self._splitter = Splitter3D(self._mesh)
        self.setCentralWidget(self._splitter)

    def _add_menubar(self):
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu('File')
        file_menu.addAction(self._create_action('open...', self.on_file_open))

    def _create_action(self, name: str, handler: Callable[[], None]) -> QAction:
        action = QAction(name, self)
        action.triggered.connect(handler)
        return action

    @staticmethod
    def _read_mesh(stl_path: str) -> trimesh.Trimesh:
        mesh = trimesh.load(stl_path)
        #self._rotate_mesh_90degree_around_x_axis(mesh)
        return mesh

    def on_file_open(self) -> None:
        file_dlg = QFileDialog(self)
        file_dlg.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)
        file_dlg.setWindowTitle('Open File')

        ok = file_dlg.exec()
        if not ok:
            return


class Splitter3D(QSplitter):

    def __init__(self, mesh: trimesh.Trimesh):
        super().__init__()

        self._mesh = mesh

        self._opengl_widget = OpenGlWin(mesh=mesh)
        self._label_widget = OpenGlInfoWin(mesh=mesh)
        self._set_handlers()

        self.addWidget(self._opengl_widget)
        self.addWidget(self._label_widget)

        self.setStretchFactor(0, 1)
        self.setStretchFactor(1, 0)

    def _set_handlers(self) -> None:
        handlers = OpenGlWinHandlers(change_cur_item=self._label_widget.on_opengl_change_cur_item,
                                     change_sel_items=self._label_widget.on_opengl_change_sel_items)
        self._opengl_widget.set_handlers(handlers)


class OpenGlInfoWin(QLabel):
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

    def on_opengl_change_cur_item(self, cur_item: Optional[MeshItemKey]) -> None:
        if cur_item is None:
            label = ''
        elif cur_item.type == MeshItemType.VERTEX:
            mesh_vertex = self._mesh.vertices[cur_item.index]
            x, y, z = mesh_vertex
            label = f'vertex[{cur_item.index}]:\n({x:.3f}, {y:.3f}, {z:.3f})'
        elif cur_item.type == MeshItemType.EDGE:
            label = f'edge[{cur_item.index}]'
        elif cur_item.type == MeshItemType.FACE:
            label = f'face[{cur_item.index}]'
        else:
            label = ''

        self.setText(label)

    def on_opengl_change_sel_items(self, new_sel_items: List[MeshItemKey]) -> None:
        print(f'on_opengl_change_sel_items: {new_sel_items}')


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())