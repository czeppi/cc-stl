import sys
import trimesh
import numpy as np
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QMatrix4x4, QWindow, QVector3D
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtOpenGL import QOpenGLShaderProgram, QOpenGLBuffer, QOpenGLVertexArrayObject, QOpenGLShader
from OpenGL.GL import *


VERTEX_SHADER_SRC = """
    #version 330 core
    layout(location = 0) in vec3 position;
    layout(location = 1) in vec3 normal; 

    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;

    out vec3 fragNormal;
    out vec3 fragPosition;

    void main() {
        // fragPosition = vec3(model * vec4(position, 1.0));
        // fragNormal = mat3(transpose(inverse(model))) * normal;
        fragPosition = position;
        fragNormal = vec3(normal);
        
        gl_Position = projection * view * vec4(fragPosition, 1.0);
    }
"""

FRAGMENT_SHADER_SRC = """
    #version 330 core
    in vec3 fragNormal;
    in vec3 fragPosition;

    uniform vec3 ambientColor;    // Umgebungslichtfarbe
    uniform vec3 cameraPos;
    uniform vec3 objectColor; 

    out vec4 fragColor;

    void main() {
        vec3 normal = normalize(fragNormal);
        vec3 viewDir = normalize(cameraPos - fragPosition);
        float intensity = max(dot(viewDir, normal), 0.0);
        vec3 result = ambientColor * objectColor * intensity + 0.2;  // todo: offset is nor correct

        fragColor = vec4(result, 1.0);
    }
"""


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
        self._camera_distance = 70.0  # start zoom distance
        self._camera_azimuth = 0.0  # horizontal angle in degree
        self._camera_elevation = 0.0  # vertical angle in degree
        self._last_mouse_position = QPoint()
        self._is_right_button_pressed = False

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)

        # read STL file
        self._mesh = trimesh.load(self._stl_path)
        vertices = np.array(self._mesh.vertices, dtype=np.float32)
        #vertices[:, [1, 2]] = vertices[:, [2, 1]]   # swap y and z, cause z should vertical axis
        faces = np.array(self._mesh.faces, dtype=np.uint32)

        vertex_normals = np.array(self._mesh.vertex_normals, dtype=np.float32)
        face_normals = np.array(self._mesh.face_normals, dtype=np.float32)

        # combine positions and vertex_normals
        vertex_data = np.hstack((vertices, vertex_normals)).astype(np.float32)

        # variant for flat shading (homogenous colored triangles)
        #face_normals = np.repeat(face_normals, 3, axis=0)  # repeat for every vertex per face
        #faces_flatten = faces.flatten()
        #vertex_data = np.hstack((vertices[faces.flatten()], face_normals)).astype(np.float32)  # for face normals instead of vertex normals

        self._shader_program = QOpenGLShaderProgram(self)
        self._shader_program.addShaderFromSourceCode(QOpenGLShader.Vertex, VERTEX_SHADER_SRC)
        self._shader_program.addShaderFromSourceCode(QOpenGLShader.Fragment, FRAGMENT_SHADER_SRC)
        self._shader_program.link()

        # create and init VBO und EBO
        self._vbo = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
        self._vbo.create()
        self._vbo.bind()
        self._vbo.allocate(vertex_data.tobytes(), vertex_data.nbytes)

        self._ebo = QOpenGLBuffer(QOpenGLBuffer.IndexBuffer)
        self._ebo.create()
        self._ebo.bind()
        self._ebo.allocate(faces.tobytes(), faces.nbytes)

        # init VAO
        self._vao = QOpenGLVertexArrayObject()
        self._vao.create()
        self._vao.bind()
        self._vbo.bind()
        self._ebo.bind()
        self._shader_program.bind()

        pos_location = self._shader_program.attributeLocation("position")
        norm_location = self._shader_program.attributeLocation("normal")
        self._shader_program.enableAttributeArray(pos_location)
        self._shader_program.enableAttributeArray(norm_location)
        glVertexAttribPointer(pos_location, 3, GL_FLOAT, GL_FALSE, 6 * vertex_data.itemsize, ctypes.c_void_p(0))
        glVertexAttribPointer(norm_location, 3, GL_FLOAT, GL_FALSE, 6 * vertex_data.itemsize, ctypes.c_void_p(3 * vertex_data.itemsize))

        # set light parameters
        self._shader_program.setUniformValue("ambientColor", QVector3D(0.5, 0.5, 0.5))  # Gedimmtes Umgebungslicht
        self._shader_program.setUniformValue("objectColor", QVector3D(0.6, 0.6, 0.8))  # object color

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
        model_matrix = self._create_eye_matrix()
        view_matrix = self._calculate_view_matrix()

        # calc camera position
        elevation_rad = np.radians(self._camera_elevation)
        azimuth_rad = np.radians(self._camera_azimuth)
        camera_x = self._camera_distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
        camera_y = self._camera_distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
        camera_z = self._camera_distance * np.sin(elevation_rad)

        # set init shader program
        self._shader_program.setUniformValue("model", model_matrix)
        self._shader_program.setUniformValue("view", view_matrix)
        self._shader_program.setUniformValue("cameraPos", QVector3D(camera_x, camera_y, camera_z))

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

    def _calculate_view_matrix(self) -> QMatrix4x4:
        """ cals view matrix in dependency of the camera
        """
        # calc camera position with azimuth and elevation
        x = self._camera_distance * np.cos(np.radians(self._camera_elevation)) * np.sin(np.radians(self._camera_azimuth))
        y = self._camera_distance * np.cos(np.radians(self._camera_elevation)) * np.cos(np.radians(self._camera_azimuth))
        z = self._camera_distance * np.sin(np.radians(self._camera_elevation))

        # camera view point and "up"-vector
        eye = np.array([x, y, z], dtype=np.float32)
        center = np.array([0, 0, 0], dtype=np.float32)
        up = np.array([0, 0, 1], dtype=np.float32)

        # View-Matrix berechnen
        f = center - eye
        f /= np.linalg.norm(f)
        s = np.cross(up, f)
        s /= np.linalg.norm(s)
        u = np.cross(f, s)

        return QMatrix4x4(
            s[0], u[0], -f[0], -np.dot(s, eye),
            s[1], u[1], -f[1], -np.dot(u, eye),
            s[2], u[2], -f[2], np.dot(f, eye),
            0, 0, 0, 1
        )

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
            self._camera_azimuth += delta.x() * 0.5  # sensitivity for azimuth
            self._camera_elevation += delta.y() * 0.5  # sensitivity for elevation

            # bound elevation
            self._camera_elevation = max(-89.0, min(89.0, self._camera_elevation))

            self._last_mouse_position = event.position().toPoint()
            self.update()  # update screen

    def wheelEvent(self, event):
        # adapt zoom-factor (p.e., 1.1 for fast zoom or 1.05 for slow zoom)
        zoom_factor = 0.9 if event.angleDelta().y() > 0 else 1.1
        self._camera_distance *= zoom_factor
        self._camera_distance = max(1.0, min(100.0, self._camera_distance))  # bound zoom interval
        self.update()  # update screen


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