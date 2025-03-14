from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Callable, List

import numpy as np
import trimesh
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMainWindow, QSplitter, QFileDialog, QMessageBox

from meshinfowin import MeshInfoWin, MeshInfoWinHandlers
from openglwidget import GL_VIEW_SIZE, OpenGlWin, OpenGlWinHandlers

ROOT_DPATH = Path(sys.argv[0]).absolute().parent.parent
STL_DPATH = ROOT_DPATH / "stl-files"
STL_FPATH = STL_DPATH / "KLP_Lame_Tilted.stl"
#STL_FPATH = STL_DPATH / "charybdisnano_v2_v187.stl"
#STL_FPATH = STL_DPATH / "adapter_v2_bottom_pmw_3389.stl"


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("STL Example")
        self.resize(*GL_VIEW_SIZE)

        self._mesh_orig = self._read_mesh(STL_FPATH)
        self._mesh_moved = self._mesh_orig.copy()
        self._mesh_with_plane = self._add_xy_plane(self._mesh_moved)

        self._add_menubar()

        self._splitter = Splitter3D(self._mesh_with_plane)
        self.setCentralWidget(self._splitter)

    def _add_menubar(self):
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu('File')
        file_menu.addAction(self._create_action('open...', self.on_file_open))

        mesh_menu = menu_bar.addMenu('Mesh')
        mesh_menu.addAction(self._create_action('rotate x', self.on_mesh_rotate_x))
        mesh_menu.addAction(self._create_action('rotate y', self.on_mesh_rotate_y))
        mesh_menu.addAction(self._create_action('rotate z', self.on_mesh_rotate_z))
        mesh_menu.addAction(self._create_action('adjust z', self.on_mesh_adjust_xy_plane))
        mesh_menu.addAction(self._create_action('restore to origin', self.on_mesh_restore_to_origin))

        help_menu  = menu_bar.addMenu('Help')
        help_menu.addAction(self._create_action('About', self.on_help_about))

    def _create_action(self, name: str, handler: Callable[[], None]) -> QAction:
        action = QAction(name, self)
        action.triggered.connect(handler)
        return action

    @staticmethod
    def _read_mesh(stl_path: Path) -> trimesh.Trimesh:
        mesh = trimesh.load(stl_path)
        #self._rotate_mesh_90degree_around_x_axis(mesh)
        return mesh

    @staticmethod
    def _add_xy_plane(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        bbox = mesh.bounds
        x_min, y_min, z_min = bbox[0]
        x_max, y_max, z_max = bbox[1]
        border = min(abs(x_max - x_min), abs(y_max - y_min)) / 2
        x_min -= border
        x_max += border
        y_min -= border
        y_max += border
        xy_vertices = np.array([
            [x_min, y_min, z_min],
            [x_max, y_min, z_min],
            [x_max, y_max, z_min],
            [x_min, y_max, z_min],
        ])
        xy_faces = np.array([
            [0, 1, 2],
            [0, 2, 3],
        ])
        xy_plane = trimesh.Trimesh(vertices=xy_vertices, faces=xy_faces)
        combined_mesh = trimesh.util.concatenate(mesh, xy_plane)
        return combined_mesh

    def on_file_open(self) -> None:
        fpath_str, _ = QFileDialog.getOpenFileName(self, 'Open File', str(STL_DPATH))
        if not fpath_str:
            return

        fpath = Path(fpath_str)
        if not fpath.exists():
            return

        self._mesh_orig = self._read_mesh(fpath)
        self._mesh_moved = self._mesh_orig.copy()
        self._mesh_with_plane = self._add_xy_plane(self._mesh_moved)

        self._splitter = Splitter3D(self._mesh_with_plane)
        self.setCentralWidget(self._splitter)

    def on_mesh_rotate_x(self) -> None:
        self._rotate_mesh_90_degree(direction=[1, 0, 0])

    def on_mesh_rotate_y(self) -> None:
        self._rotate_mesh_90_degree(direction=[0, 1, 0])

    def on_mesh_rotate_z(self) -> None:
        self._rotate_mesh_90_degree(direction=[0, 0, 1])

    def on_mesh_adjust_xy_plane(self) -> None:
        pass

    def on_mesh_restore_to_origin(self) -> None:
        self._mesh_moved = self._mesh_orig.copy()
        self._mesh_with_plane = self._add_xy_plane(self._mesh_moved)
        self._splitter = Splitter3D(self._mesh_with_plane)
        self.setCentralWidget(self._splitter)

    def _rotate_mesh_90_degree(self, direction: List[float]) -> None:
        rotation_matrix = trimesh.transformations.rotation_matrix(math.pi / 2, direction)
        self._mesh_moved.apply_transform(rotation_matrix)
        self._mesh_with_plane = self._add_xy_plane(self._mesh_moved)
        self._splitter = Splitter3D(self._mesh_with_plane)
        self.setCentralWidget(self._splitter)

    def on_help_about(self) -> None:
        # QMessageBox.aboutQt(self, 'title')
        lines = [
            'CC-STL',
            '',
            'Copyright(C) 2024 Christian Czepluch',
            '',
            'Author: Christian Czepluch',
        ]
        QMessageBox.information(self, 'About cc-stl', '\n'.join(lines))


class Splitter3D(QSplitter):

    def __init__(self, mesh: trimesh.Trimesh):
        super().__init__()

        self._opengl_widget = OpenGlWin(mesh=mesh)
        self._mesh_info_win = MeshInfoWin(mesh=mesh)
        self._set_handlers()

        self.addWidget(self._opengl_widget)
        self.addWidget(self._mesh_info_win)

        self.setSizes([1, 300])

        self.setStretchFactor(0, 1)
        self.setStretchFactor(1, 0)

    def _set_handlers(self) -> None:
        opengl_handlers = OpenGlWinHandlers(change_camera_pos=self._mesh_info_win.on_opengl_change_camera_pos,
                                            change_cur_item=self._mesh_info_win.on_opengl_change_cur_item,
                                            change_sel_items=self._mesh_info_win.on_opengl_change_sel_items)
        self._opengl_widget.set_handlers(opengl_handlers)

        mesh_info_handlers = MeshInfoWinHandlers(
            on_global_analyze_complete=self._opengl_widget.on_global_analyze_complete)
        self._mesh_info_win.set_handlers(mesh_info_handlers)
