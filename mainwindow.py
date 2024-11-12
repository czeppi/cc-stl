from __future__ import annotations

import sys
from typing import Callable, List

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

        self._opengl_widget = self._create_opengl_widget(self._mesh)
        self._label_widget = QLabel(parent=self)
        self._label_widget.setText('hallo')

        self.addWidget(self._opengl_widget)
        self.addWidget(self._label_widget)

        self.setStretchFactor(0, 1)
        self.setStretchFactor(1, 0)

    def _create_opengl_widget(self, mesh: trimesh.Trimesh) -> OpenGlWin:
        handlers = OpenGlWinHandlers(change_cur_item=self.on_opengl_change_cur_item,
                                     change_sel_items=self.on_opengl_change_sel_items)
        return OpenGlWin(mesh=mesh, handlers=handlers)

    def on_opengl_change_cur_item(self, cur_item: MeshItemKey) -> None:
        print(f'on_opengl_change_cur_item: {cur_item}')

    def on_opengl_change_sel_items(self, new_sel_items: List[MeshItemKey]) -> None:
        print(f'on_opengl_change_sel_items: {new_sel_items}')


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())