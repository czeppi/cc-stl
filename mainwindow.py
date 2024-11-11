from __future__ import annotations

import sys

import trimesh
from PySide6.QtWidgets import QMainWindow, QApplication

from openglwidget import GL_VIEW_SIZE, OpenGlWin


STL_PATH = "stl-files/KLP_Lame_Tilted.stl"
#STL_PATH = "stl-files/charybdisnano_v2_v187.stl"
#STL_PATH = "stl-files/adapter_v2_bottom_pmw_3389.stl"


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("STL Example")
        #self.setFixedSize(*GL_VIEW_SIZE)
        self.resize(*GL_VIEW_SIZE)

        self._mesh = self._read_mesh(STL_PATH)
        self._open_gl_widget = OpenGlWin(mesh= self._mesh, parent=self)
        self.setCentralWidget(self._open_gl_widget)

    @staticmethod
    def _read_mesh(stl_path: str) -> trimesh.Trimesh:
        mesh = trimesh.load(stl_path)
        #self._rotate_mesh_90degree_around_x_axis(mesh)
        return mesh


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())