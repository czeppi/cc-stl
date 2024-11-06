import sys
import trimesh
import numpy as np
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QMatrix4x4, QWindow, QVector3D
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtOpenGL import QOpenGLShaderProgram, QOpenGLBuffer, QOpenGLVertexArrayObject, QOpenGLShader
from OpenGL.GL import *


FACES_VERTEX_SHADER_SRC = """
    #version 330 core
    layout(location = 0) in vec3 position;
    layout(location = 1) in vec3 normal; 

    uniform mat4 mvp_matrix;

    out vec3 fragNormal;
    out vec3 fragPosition;

    void main() {
        // fragPosition = vec3(model * vec4(position, 1.0));
        // fragNormal = mat3(transpose(inverse(model))) * normal;
        fragPosition = position;
        fragNormal = vec3(normal);
        
        gl_Position = mvp_matrix * vec4(fragPosition, 1.0);
    }
"""

FACES_FRAGMENT_SHADER_SRC = """
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

        fragColor = vec4(result, 0.5);
    }
"""


EDGES_VERTEX_SHADER_SRC = """
    #version 330
    layout(location = 0) in vec3 position;
    uniform mat4 mvp_matrix;
    void main() {
        gl_Position = mvp_matrix * vec4(position, 1.0);
    }
"""


EDGES_FRAGMENT_SHADER_SRC = """
    #version 330
    out vec4 fragColor;
    void main() {
        fragColor = vec4(0.0, 0.0, 0.0, 1.0);  // blck
    }
"""


# not used
VERTICES_VERTEX_SHADER_SRC = """
    #version 330
    layout(location = 0) in vec3 position;
    uniform mat4 mvp_matrix;
    void main() {
        gl_Position = mvp_matrix * vec4(position, 1.0);
    }
"""


# not used
VERTICES_FRAGMENT_SHADER_SRC = """
    #version 330
    out vec4 fragColor;
    void main() {
        fragColor = vec4(0.0, 0.0, 0.0, 1.0);  // selected color for vertices
    }
"""


class OpenGlWin(QOpenGLWidget):

    def __init__(self, stl_path: str, parent: QWindow):
        super().__init__()
        self._mesh = self._read_mesh(stl_path)
        self._edges_array = np.array(self._mesh.edges_unique, dtype=np.uint32)  # self._mesh.edges_unique

        self._faces_shader_program = QOpenGLShaderProgram()
        self._faces_vbo = None
        self._faces_ebo = None
        self._vao = None

        self._projection_matrix = QMatrix4x4()

        # Kamera-Parameter
        self._camera_distance = 50.0  # start zoom distance
        self._camera_azimuth = 0.0  # horizontal angle in degree
        self._camera_elevation = 0.0  # vertical angle in degree
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
        self._faces_shader_program = self._create_faces_shader_program()
        self._edges_shader_program = self._create_edges_shader_program()
        self._vertices_shader_program = self._create_vertices_shader_program()

        vertices_data = self._create_vertices_data(self._mesh)

        self._faces_vbo = self._create_faces_vbo_buffer(vertices_data)
        self._faces_ebo = self._create_faces_ebo_buffer(self._mesh)
        self._edges_ebe = self._create_edges_ebo_buffer(self._edges_array)

        # init VAO
        self._vao = QOpenGLVertexArrayObject()
        self._vao.create()
        self._vao.bind()
        self._faces_vbo.bind()
        self._faces_ebo.bind()
        self._faces_shader_program.bind()

        self._init_faces_shader_program(self._faces_shader_program, vertices_data.itemsize)

        self._vao.release()
        self._faces_vbo.release()
        self._faces_ebo.release()
        self._faces_shader_program.release()

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def _create_faces_shader_program(self) -> QOpenGLShaderProgram:
        shader_program = QOpenGLShaderProgram(self)
        shader_program.addShaderFromSourceCode(QOpenGLShader.Vertex, FACES_VERTEX_SHADER_SRC)
        shader_program.addShaderFromSourceCode(QOpenGLShader.Fragment, FACES_FRAGMENT_SHADER_SRC)
        shader_program.link()
        return shader_program

    def _create_edges_shader_program(self)-> QOpenGLShaderProgram:
        shader_program = QOpenGLShaderProgram(self)
        shader_program.addShaderFromSourceCode(QOpenGLShader.Vertex, EDGES_VERTEX_SHADER_SRC)
        shader_program.addShaderFromSourceCode(QOpenGLShader.Fragment, EDGES_FRAGMENT_SHADER_SRC)
        shader_program.link()
        return shader_program

    def _create_vertices_shader_program(self)-> QOpenGLShaderProgram:
        shader_program = QOpenGLShaderProgram(self)
        shader_program.addShaderFromSourceCode(QOpenGLShader.Vertex, VERTICES_VERTEX_SHADER_SRC)
        shader_program.addShaderFromSourceCode(QOpenGLShader.Fragment, VERTICES_FRAGMENT_SHADER_SRC)
        shader_program.link()
        return shader_program

    @staticmethod
    def _create_vertices_data(mesh: trimesh.Trimesh) -> np.array:
        vertices_array = np.array(mesh.vertices, dtype=np.float32)
        # vertices_array[:, [1, 2]] = vertices_array[:, [2, 1]]   # swap y and z, cause z should vertical axis

        vertex_normals = np.array(mesh.vertex_normals, dtype=np.float32)
        vertices_data = np.hstack((vertices_array, vertex_normals)).astype(np.float32)

        # # variant for flat shading (homogenous colored triangles)
        # face_normals = np.array(self._mesh.face_normals, dtype=np.float32)
        # face_normals = np.repeat(face_normals, 3, axis=0)  # repeat for every vertex per face
        # faces_flatten = faces.flatten()
        # vertices_data = np.hstack((vertices_array[faces.flatten()], face_normals)).astype(np.float32)  # for face normals instead of vertex normals

        return vertices_data

    @staticmethod
    def _init_faces_shader_program(shader_program: QOpenGLShaderProgram, item_size: int) -> None:
        pos_location = shader_program.attributeLocation("position")
        norm_location = shader_program.attributeLocation("normal")
        shader_program.enableAttributeArray(pos_location)
        shader_program.enableAttributeArray(norm_location)
        glVertexAttribPointer(pos_location, 3, GL_FLOAT, GL_FALSE, 6 * item_size, ctypes.c_void_p(0))
        glVertexAttribPointer(norm_location, 3, GL_FLOAT, GL_FALSE, 6 * item_size, ctypes.c_void_p(3 * item_size))

        # set light parameters
        shader_program.setUniformValue("ambientColor", QVector3D(0.5, 0.5, 0.5))  # Gedimmtes Umgebungslicht
        shader_program.setUniformValue("objectColor", QVector3D(0.6, 0.6, 0.8))  # object color

    @staticmethod
    def _create_faces_vbo_buffer(vertices_data: np.array) -> QOpenGLBuffer:
        vbo = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
        vbo.create()
        vbo.bind()
        vbo.setUsagePattern(QOpenGLBuffer.StaticDraw)
        vbo.allocate(vertices_data.tobytes(), vertices_data.nbytes)
        return vbo

    @staticmethod
    def _create_faces_ebo_buffer(mesh: trimesh.Trimesh) -> QOpenGLBuffer:
        faces = np.array(mesh.faces, dtype=np.uint32)

        ebo = QOpenGLBuffer(QOpenGLBuffer.IndexBuffer)
        ebo.create()
        ebo.bind()
        ebo.setUsagePattern(QOpenGLBuffer.StaticDraw)
        ebo.allocate(faces.tobytes(), faces.nbytes)
        return ebo

    @staticmethod
    def _create_edges_ebo_buffer(edges_array: np.array) -> QOpenGLBuffer:
        ebo = QOpenGLBuffer(QOpenGLBuffer.IndexBuffer)
        ebo.create()
        ebo.bind()
        ebo.setUsagePattern(QOpenGLBuffer.StaticDraw)
        ebo.allocate(edges_array.tobytes(), edges_array.nbytes)
        return ebo

    def resizeGL(self, width, height):
        glViewport(0, 0, width, height)

        aspect_ratio = width / height
        self._projection_matrix = self._create_perspective_matrix(45.0, aspect_ratio, 0.1, 100.0)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # mvp_matrix
        model_matrix = self._create_eye_matrix()
        view_matrix = self._calculate_view_matrix()
        mvp_matrix = self._projection_matrix * view_matrix * model_matrix

        # calc camera position
        elevation_rad = np.radians(self._camera_elevation)
        azimuth_rad = np.radians(self._camera_azimuth)
        camera_x = self._camera_distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
        camera_y = self._camera_distance * np.sin(elevation_rad)
        camera_z = self._camera_distance * np.cos(elevation_rad) * np.cos(azimuth_rad)

        # paint faces
        self._faces_shader_program.bind()
        self._faces_shader_program.setUniformValue("mvp_matrix", mvp_matrix)
        self._faces_shader_program.setUniformValue("cameraPos", QVector3D(camera_x, camera_y, camera_z))
        self._vao.bind()

        glDrawElements(GL_TRIANGLES, self._mesh.faces.size, GL_UNSIGNED_INT, None)

        self._vao.release()
        self._faces_shader_program.release()

        # paint edges
        glDepthMask(GL_FALSE)  # depth buffer writing deactivate
        glLineWidth(2.0)

        self._edges_shader_program.bind()
        self._edges_shader_program.setUniformValue("mvp_matrix", mvp_matrix)
        self._vao.bind()

        glDrawElements(GL_LINES, self._edges_array.size, GL_UNSIGNED_INT, None)

        self._vao.release()
        self._edges_shader_program.release()
        glDepthMask(GL_TRUE)  # depth buffer writing reactivate

        # paint vertices
        glPointSize(5.0)
        self._vertices_shader_program.bind()
        self._vertices_shader_program.setUniformValue("mvp_matrix", mvp_matrix)
        self._vao.bind()

        glDrawElements(GL_POINTS, self._mesh.vertices.size, GL_UNSIGNED_INT, None)

        self._vao.release()
        self._edges_shader_program.release()

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
        y = self._camera_distance * np.sin(np.radians(self._camera_elevation))
        z = self._camera_distance * np.cos(np.radians(self._camera_elevation)) * np.cos(np.radians(self._camera_azimuth))

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

        return QMatrix4x4(
            s[0], u[0], f[0], -np.dot(s, eye),
            s[1], u[1], f[1], -np.dot(u, eye),
            s[2], u[2], f[2], np.dot(f, eye),
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
            self._camera_azimuth += delta.x() * 0.25  # sensitivity for azimuth
            self._camera_elevation += delta.y() * 0.25  # sensitivity for elevation

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
        self.setFixedSize(1200, 900)
        self._open_gl_widget = OpenGlWin(stl_path="KLP_Lame_Tilted.stl", parent=self)
        self.setCentralWidget(self._open_gl_widget)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())