from __future__ import annotations

import sys
from typing import Optional

import numpy as np
import trimesh
from OpenGL.GL import *
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QMatrix4x4, QWindow
from PySide6.QtOpenGL import QOpenGLBuffer
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QApplication, QMainWindow

from camera import Camera
from shaders import FacesShaderProgram, EdgesShaderProgram, VerticesShaderProgram


class OpenGlWin(QOpenGLWidget):

    def __init__(self, stl_path: str, parent: QWindow):
        super().__init__()
        self._mesh = self._read_mesh(stl_path)
        self._edges_array = np.array(self._mesh.edges_unique, dtype=np.uint32)  # self._mesh.edges_unique

        self._faces_shader_program = FacesShaderProgram(self._mesh)
        self._edges_shader_program = EdgesShaderProgram(self._mesh)
        self._vertices_shader_program = VerticesShaderProgram(self._mesh)

        self._positions_vbo: Optional[QOpenGLBuffer] = None
        self._normals_vbo: Optional[QOpenGLBuffer] = None

        self._projection_matrix = QMatrix4x4()

        # camera
        self._camera = Camera(distance=50.0, azimuth=180.0, elevation=45.0)
        self._last_mouse_position = QPoint()
        self._is_right_button_pressed = False

    def _read_mesh(self, stl_path: str) -> trimesh.Trimesh:
        mesh = trimesh.load(stl_path)
        self._rotate_mesh_90degree_around_x_axis(mesh)
        return mesh

    @staticmethod
    def _rotate_mesh_90degree_around_x_axis(mesh: trimesh.Trimesh) -> None:
        """ rotate mesh, so that z axis looks up instead of y axis
        """
        rot_mat = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
        mesh.apply_transform(rot_mat)

    def initializeGL(self):
        self._faces_shader_program.link()
        self._edges_shader_program.link()
        self._vertices_shader_program.link()

        self._positions_vbo = self._create_positions_vbo_buffer(np.array(self._mesh.vertices, dtype=np.float32))
        self._normals_vbo = self._create_normals_vbo_buffer(np.array(self._mesh.vertex_normals, dtype=np.float32))

        self._faces_shader_program.init(positions_vbo=self._positions_vbo, normals_vbo=self._normals_vbo)
        self._edges_shader_program.init(positions_vbo=self._positions_vbo)
        self._vertices_shader_program.init(positions_vbo=self._positions_vbo)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    @staticmethod
    def _create_positions_vbo_buffer(vertices: np.array) -> QOpenGLBuffer:
        vbo = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
        vbo.create()
        vbo.bind()
        vbo.setUsagePattern(QOpenGLBuffer.StaticDraw)
        vbo.allocate(vertices.tobytes(), vertices.nbytes)
        vbo.release()
        return vbo

    @staticmethod
    def _create_normals_vbo_buffer(vertices: np.array) -> QOpenGLBuffer:
        vbo = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
        vbo.create()
        vbo.bind()
        vbo.setUsagePattern(QOpenGLBuffer.StaticDraw)
        vbo.allocate(vertices.tobytes(), vertices.nbytes)
        vbo.release()
        return vbo

    def resizeGL(self, width, height):
        glViewport(0, 0, width, height)

        aspect_ratio = width / height
        self._projection_matrix = self._create_perspective_matrix(45.0, aspect_ratio, 0.1, 100.0)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # mvp_matrix
        model_matrix = self._create_eye_matrix()
        view_matrix = self._calculate_view_matrix()
        view_matrix2 = self._calculate_view_matrix_new()
        print(f'xyz={self._camera.xyz}')
        print(f'    old={self._mat4_str(view_matrix)}')
        print(f'    new={self._mat4_str(view_matrix2)}')

        mvp_matrix = self._projection_matrix * view_matrix * model_matrix

        # paint
        self._faces_shader_program.paint(camera=self._camera, mvp_matrix=mvp_matrix)
        self._edges_shader_program.paint(mvp_matrix=mvp_matrix)
        #self._vertices_shader_program.paint(mvp_matrix=mvp_matrix)

    def _mat4_str(self, m):
        parts = [f'{v:.03f}' for v in self._mat4_items(m)]
        return ', '.join(parts)

    def _mat4_items(self, m):
        yield m[0,0]
        yield m[0,1]
        yield m[0,2]
        yield m[0,3]
        yield m[1,0]
        yield m[1,1]
        yield m[1,2]
        yield m[1,3]
        yield m[2,0]
        yield m[2,1]
        yield m[2,2]
        yield m[2,3]
        yield m[3,0]
        yield m[3,1]
        yield m[3,2]
        yield m[3,3]

    @staticmethod
    def _create_eye_matrix() -> QMatrix4x4:
        return QMatrix4x4(
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1
        )

    @staticmethod
    def _create_perspective_matrix(fov, aspect, near, far) -> QMatrix4x4:
        f = 1.0 / np.tan(np.radians(fov) / 2)
        return QMatrix4x4(
            f / aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, (far + near) / (near - far), (2 * far * near) / (near - far),
            0, 0, -1, 0
        )

    @staticmethod
    def _create_translation_matrix(x, y, z) -> QMatrix4x4:
        return QMatrix4x4(
            1, 0, 0, x,
            0, 1, 0, y,
            0, 0, 1, z,
            0, 0, 0, 1
        )

    def _calculate_view_matrix_new(self) -> QMatrix4x4:
        """ cals view matrix in dependency of the camera
        """
        eye = np.array(self._camera.xyz, dtype=np.float32)
        up_vector = np.array([0, 1, 0], dtype=np.float32)

        z_axis = 1 * eye
        z_axis /= np.linalg.norm(z_axis)

        x_axis = np.cross(up_vector, z_axis)
        x_axis /= np.linalg.norm(x_axis)

        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)

        view_matrix = np.identity(4)
        view_matrix[0, :3] = x_axis
        view_matrix[1, :3] = y_axis
        view_matrix[2, :3] = z_axis
        view_matrix[:3, 3] = -eye @ np.array([x_axis, y_axis, z_axis])

        view_matrix_flatten = view_matrix.flatten()
        m = QMatrix4x4(*view_matrix_flatten)
        return m

    def _calculate_view_matrix(self) -> QMatrix4x4:
        """ cals view matrix in dependency of the camera
        """
        eye = np.array(self._camera.xyz, dtype=np.float32)
        up_vector = np.array([0, 1, 0], dtype=np.float32)

        z_axis = 1 * eye
        z_axis /= np.linalg.norm(z_axis)

        x_axis = np.cross(up_vector, z_axis)
        x_axis /= np.linalg.norm(x_axis)

        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)

        # m = QMatrix4x4(
        #     x_axis[0], y_axis[0], z_axis[0], -np.dot(x_axis, eye),
        #     x_axis[1], y_axis[1], z_axis[1], -np.dot(y_axis, eye),
        #     x_axis[2], y_axis[2], z_axis[2], -np.dot(z_axis, eye),
        #     0, 0, 0, 1
        # )
        m = QMatrix4x4(
            x_axis[0], x_axis[1], x_axis[2], -np.dot(x_axis, eye),
            y_axis[0], y_axis[1], y_axis[2], -np.dot(y_axis, eye),
            z_axis[0], z_axis[1], z_axis[2], -np.dot(z_axis, eye),
            0, 0, 0, 1
        )
        return m

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self._is_right_button_pressed = True
            self._last_mouse_position = event.position().toPoint()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton:
            self._is_right_button_pressed = False

    def mouseMoveEvent(self, event):
        if self._is_right_button_pressed:
            delta = event.position().toPoint() - self._last_mouse_position
            d_azimuth = -delta.x() * 0.25  # sensitivity for azimuth
            d_elevation = delta.y() * 0.25  # sensitivity for elevation

            self._camera.azimuth += d_azimuth
            #self._camera.rotate_vertical(d_elevation)
            self._camera.elevation += d_elevation
            x, y, z = self._camera.xyz
            #print(f'elevation: {self._camera._elevation}, y: {y}')

            self._last_mouse_position = event.position().toPoint()
            self.update()  # update screen

    def wheelEvent(self, event):
        # adapt zoom-factor (p.e., 1.1 for fast zoom or 1.05 for slow zoom)
        zoom_factor = 0.9 if event.angleDelta().y() > 0 else 1.1
        self._camera.distance *= zoom_factor
        self.update()  # update screen


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("STL Example")
        self.setFixedSize(1200, 900)
        self._open_gl_widget = OpenGlWin(stl_path="KLP_Lame_Tilted.stl", parent=self)
        self.setCentralWidget(self._open_gl_widget)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())