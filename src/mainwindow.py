from __future__ import annotations

import sys
from typing import Callable

import trimesh
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMainWindow, QApplication, QSplitter, QFileDialog

from meshinfowin import MeshInfoWin
from openglwidget import GL_VIEW_SIZE, OpenGlWin, OpenGlWinHandlers

STL_DPATH = "../stl-files/"
STL_PATH = STL_DPATH + "KLP_Lame_Tilted.stl"
#STL_PATH = STL_DPATH + "charybdisnano_v2_v187.stl"
#STL_PATH = STL_DPATH + "adapter_v2_bottom_pmw_3389.stl"


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

        self._opengl_widget = OpenGlWin(mesh=mesh)
        self._mesh_info_win = MeshInfoWin(mesh=mesh)
        self._set_handlers()

        self.addWidget(self._opengl_widget)
        self.addWidget(self._mesh_info_win)

        self.setSizes([1, 200])

        self.setStretchFactor(0, 1)
        self.setStretchFactor(1, 0)

    def _set_handlers(self) -> None:
        handlers = OpenGlWinHandlers(change_camera_pos=self._mesh_info_win.on_opengl_change_camera_pos,
                                     change_cur_item=self._mesh_info_win.on_opengl_change_cur_item,
                                     change_sel_items=self._mesh_info_win.on_opengl_change_sel_items)
        self._opengl_widget.set_handlers(handlers)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())
