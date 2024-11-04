import sys
import trimesh
import numpy as np
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QMatrix4x4, QWindow
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtOpenGL import QOpenGLShaderProgram, QOpenGLBuffer, QOpenGLVertexArrayObject, QOpenGLShader
from OpenGL.GL import *


class OpenGlWin(QOpenGLWidget):

    def __init__(self, stl_path, parent: QWindow):
        super().__init__()
        self._stl_path = stl_path
        self._mesh = None
        self._shader_program = None
        self._vbo = None
        self._ebo = None
        self._vao = None

        # Kamera-Parameter
        self.camera_distance = 50.0  # Anfangs-Zoom-Distanz
        self.camera_azimuth = 0.0  # Winkel um die y-Achse (horizontaler Winkel)
        self.camera_elevation = 0.0  # Winkel um die x-Achse (vertikaler Winkel)
        self.last_mouse_position = QPoint()
        self.is_right_button_pressed = False

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)

        # read STL file
        self._mesh = trimesh.load(self._stl_path)
        vertices = np.array(self._mesh.vertices, dtype=np.float32)
        faces = np.array(self._mesh.faces, dtype=np.uint32)

        # create shader-program
        vertex_src = """
            #version 330 core
            layout(location = 0) in vec3 position;
            uniform mat4 model;
            uniform mat4 view;
            uniform mat4 projection;
            void main() {
                gl_Position = projection * view * model * vec4(position, 1.0);
            }
        """

        fragment_src = """
            #version 330 core
            out vec4 fragColor;
            void main() {
                fragColor = vec4(0.6, 0.6, 0.8, 1.0);
            }
        """
        self._shader_program = QOpenGLShaderProgram(self)
        self._shader_program.addShaderFromSourceCode(QOpenGLShader.Vertex, vertex_src)
        self._shader_program.addShaderFromSourceCode(QOpenGLShader.Fragment, fragment_src)
        self._shader_program.link()

        # create and init VBO und EBO
        self._vbo = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
        self._vbo.create()
        self._vbo.bind()
        vertices_as_bytes = vertices.tobytes()
        self._vbo.allocate(vertices_as_bytes, len(vertices_as_bytes))

        self._ebo = QOpenGLBuffer(QOpenGLBuffer.IndexBuffer)
        self._ebo.create()
        self._ebo.bind()
        faces_as_bytes = faces.tobytes()
        self._ebo.allocate(faces_as_bytes, len(faces_as_bytes))

        # init VAO
        self._vao = QOpenGLVertexArrayObject()
        self._vao.create()
        self._vao.bind()
        self._vbo.bind()
        self._ebo.bind()
        self._shader_program.bind()

        pos_location = self._shader_program.attributeLocation("position")
        self._shader_program.enableAttributeArray(pos_location)
        glVertexAttribPointer(pos_location, 3, GL_FLOAT, GL_FALSE, 0, None)

        self._vao.release()
        self._vbo.release()
        self._ebo.release()
        self._shader_program.release()

    def resizeGL(self, width, height):
        glViewport(0, 0, width, height)

        aspect_ratio = width / height
        projection_matrix = self._create_perspective_matrix(45.0, aspect_ratio, 0.1, 100.0)

        self._shader_program.bind()
        self._shader_program.setUniformValue("projection", projection_matrix)
        self._shader_program.release()

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        self._shader_program.bind()
        self._vao.bind()

        # use transformation
        #model_matrix = self._create_translation_matrix(0, 0, -30)
        model_matrix = self._create_eye_matrix()

        # view_matrix = np.eye(4, dtype=np.float32)
        # view_matrix = self._create_eye_matrix()
        view_matrix = self._calculate_view_matrix()

        self._shader_program.setUniformValue("model", model_matrix)
        self._shader_program.setUniformValue("view", view_matrix)

        glDrawElements(GL_TRIANGLES, self._mesh.faces.size, GL_UNSIGNED_INT, None)

        self._vao.release()
        self._shader_program.release()

    @staticmethod
    def _create_eye_matrix() -> QMatrix4x4:
        return QMatrix4x4(
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1
        )

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
        # return np.array([
        #     [f / aspect, 0, 0, 0],
        #     [0, f, 0, 0],
        #     [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
        #     [0, 0, -1, 0]
        # ], dtype=np.float32)

    @staticmethod
    def _create_translation_matrix(x, y, z) -> QMatrix4x4:
        return QMatrix4x4(
            1, 0, 0, x,
            0, 1, 0, y,
            0, 0, 1, z,
            0, 0, 0, 1
        )
        # return np.array([
        #     [1, 0, 0, x],
        #     [0, 1, 0, y],
        #     [0, 0, 1, z],
        #     [0, 0, 0, 1]
        # ], dtype=np.float32)

    def _calculate_view_matrix(self) -> QMatrix4x4:
        """ cals view matrix in dependency of the camera
        """
        # calc camera position with azimuth and elevation
        x = self.camera_distance * np.cos(np.radians(self.camera_elevation)) * np.sin(np.radians(self.camera_azimuth))
        y = self.camera_distance * np.sin(np.radians(self.camera_elevation))
        z = self.camera_distance * np.cos(np.radians(self.camera_elevation)) * np.cos(np.radians(self.camera_azimuth))

        # camera view point and "up"-vector
        eye = np.array([x, y, z], dtype=np.float32)
        center = np.array([0, 0, 0], dtype=np.float32)
        up = np.array([0, 1, 0], dtype=np.float32)

        # View-Matrix berechnen
        f = center - eye
        f /= np.linalg.norm(f)
        s = np.cross(up, f)
        s /= np.linalg.norm(s)
        u = np.cross(f, s)

        # view_matrix = np.eye(4, dtype=np.float32)
        # view_matrix[:3, 0] = s
        # view_matrix[:3, 1] = u
        # view_matrix[:3, 2] = -f
        # view_matrix[0, 3] = -np.dot(s, eye)
        # view_matrix[1, 3] = -np.dot(u, eye)
        # view_matrix[2, 3] = np.dot(f, eye)
        # return view_matrix

        return QMatrix4x4(
            s[0], u[0], f[0], -np.dot(s, eye),
            s[1], u[1], f[1], -np.dot(u, eye),
            s[2], u[2], f[2], np.dot(f, eye),
            0, 0, 0, 1
        )

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self.is_right_button_pressed = True
            self.last_mouse_position = event.position().toPoint()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton:
            self.is_right_button_pressed = False

    def mouseMoveEvent(self, event):
        if self.is_right_button_pressed:
            delta = event.position().toPoint() - self.last_mouse_position
            self.camera_azimuth += delta.x() * 0.5  # Sensitivität für Azimuth
            self.camera_elevation += delta.y() * 0.5  # Sensitivität für Elevation

            # Begrenze Elevation, damit der Horizont immer gerade bleibt
            self.camera_elevation = max(-89.0, min(89.0, self.camera_elevation))

            self.last_mouse_position = event.position().toPoint()
            self.update()  # Bildschirm aktualisieren

    def wheelEvent(self, event):
        # Zoom-Faktor anpassen (z.B., 1.1 für schnelles Zoomen oder 1.05 für langsameres Zoomen)
        zoom_factor = 0.9 if event.angleDelta().y() > 0 else 1.1
        self.camera_distance *= zoom_factor
        self.camera_distance = max(1.0, min(100.0, self.camera_distance))  # Begrenze den Zoombereich
        self.update()  # Bildschirm aktualisieren


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("STL Example")
        self.setFixedSize(800, 600)
        self._open_gl_widget = OpenGlWin(stl_path="KLP_Lame_Tilted.stl", parent=self)
        self.setCentralWidget(self._open_gl_widget)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())