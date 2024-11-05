import sys
import numpy as np
import trimesh
from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtGui import QMouseEvent, QWheelEvent, QSurfaceFormat, QVector3D
from PySide6.QtCore import Qt
from OpenGL import GL

VERTEX_SHADER_SRC = """
#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat3 normalMatrix;

out vec3 fragNormal;
out vec3 fragPosition;

void main() {
    fragPosition = vec3(model * vec4(position, 1.0));
    fragNormal = normalize(normalMatrix * normal);
    gl_Position = projection * view * vec4(fragPosition, 1.0);
}
"""

FRAGMENT_SHADER_SRC = """
#version 330 core
in vec3 fragNormal;
in vec3 fragPosition;

uniform vec3 ambientColor;
uniform vec3 cameraPos;
uniform vec3 objectColor;

out vec4 fragColor;

void main() {
    vec3 normal = normalize(fragNormal);
    vec3 viewDir = normalize(cameraPos - fragPosition);
    float intensity = max(dot(viewDir, normal), 0.0); // Winkelabhängigkeit

    vec3 color = ambientColor * objectColor * intensity;
    fragColor = vec4(color, 1.0);
}
"""

class OpenGLWidget(QOpenGLWidget):
    def initializeGL(self):
        # Shaderprogramm erstellen
        self.program = GL.glCreateProgram()
        vertex_shader = GL.glCreateShader(GL.GL_VERTEX_SHADER)
        GL.glShaderSource(vertex_shader, VERTEX_SHADER_SRC)
        GL.glCompileShader(vertex_shader)
        GL.glAttachShader(self.program, vertex_shader)

        fragment_shader = GL.glCreateShader(GL.GL_FRAGMENT_SHADER)
        GL.glShaderSource(fragment_shader, FRAGMENT_SHADER_SRC)
        GL.glCompileShader(fragment_shader)
        GL.glAttachShader(self.program, fragment_shader)

        GL.glLinkProgram(self.program)

        # STL-Modell laden
        self.mesh = trimesh.load("KLP_Lame_Tilted.stl")
        vertices = np.array(self.mesh.vertices, dtype=np.float32)
        normals = np.array(self.mesh.vertex_normals, dtype=np.float32)
        faces = np.array(self.mesh.faces, dtype=np.uint32)

        # Vertex-Daten vorbereiten
        vertex_data = np.hstack((vertices[faces.flatten()], normals[faces.flatten()])).astype(np.float32)

        # VAO und VBO einrichten
        self.vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao)

        self.vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL.GL_STATIC_DRAW)

        # Attribut-Pointer für Position
        position_loc = GL.glGetAttribLocation(self.program, "position")
        GL.glVertexAttribPointer(position_loc, 3, GL.GL_FLOAT, GL.GL_FALSE, 6 * vertex_data.itemsize, GL.ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(position_loc)

        # Attribut-Pointer für Normale
        normal_loc = GL.glGetAttribLocation(self.program, "normal")
        GL.glVertexAttribPointer(normal_loc, 3, GL.GL_FLOAT, GL.GL_FALSE, 6 * vertex_data.itemsize, GL.ctypes.c_void_p(3 * vertex_data.itemsize))
        GL.glEnableVertexAttribArray(normal_loc)

        # Kamera- und Transformationsparameter
        self.camera_pos = QVector3D(0.0, 0.0, 70.0)  # Startposition der Kamera, 30 Einheiten vom Ursprung entfernt
        self.camera_angle = QVector3D(0.0, 0.0, 0.0)  # Kamerawinkel
        self.zoom_factor = 1.0

        GL.glEnable(GL.GL_DEPTH_TEST)

    def resizeGL(self, width, height):
        GL.glViewport(0, 0, width, height)

    def paintGL(self):
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glUseProgram(self.program)

        # Modell-, Sicht- und Projektionsmatrix berechnen
        model = np.identity(4, dtype=np.float32)
        view = self.get_view_matrix()
        projection = self.get_projection_matrix()

        # Normalen-Matrix berechnen und setzen
        normal_matrix = np.linalg.inv(model[:3, :3]).T
        GL.glUniformMatrix3fv(GL.glGetUniformLocation(self.program, "normalMatrix"), 1, GL.GL_FALSE, normal_matrix)

        # Matrizen an den Shader übergeben
        GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.program, "model"), 1, GL.GL_FALSE, model)
        GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.program, "view"), 1, GL.GL_FALSE, view)
        GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.program, "projection"), 1, GL.GL_FALSE, projection)

        # Weitere Uniforms
        GL.glUniform3f(GL.glGetUniformLocation(self.program, "ambientColor"), 0.3, 0.3, 0.3)
        GL.glUniform3f(GL.glGetUniformLocation(self.program, "cameraPos"), *self.camera_pos.toTuple())
        GL.glUniform3f(GL.glGetUniformLocation(self.program, "objectColor"), 0.8, 0.5, 0.3)

        # Zeichnen
        GL.glBindVertexArray(self.vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, len(self.mesh.faces) * 3)
        GL.glBindVertexArray(0)

    def get_view_matrix(self):
        view = np.identity(4, dtype=np.float32)
        rotation_x = np.array([[1, 0, 0],
                               [0, np.cos(self.camera_angle.x()), -np.sin(self.camera_angle.x())],
                               [0, np.sin(self.camera_angle.x()), np.cos(self.camera_angle.x())]])
        rotation_y = np.array([[np.cos(self.camera_angle.y()), 0, np.sin(self.camera_angle.y())],
                               [0, 1, 0],
                               [-np.sin(self.camera_angle.y()), 0, np.cos(self.camera_angle.y())]])
        view[:3, :3] = rotation_y @ rotation_x
        view[:3, 3] = (-self.camera_pos).toTuple()
        return view

    def get_projection_matrix(self):
        fov = np.radians(45)
        aspect = self.width() / self.height()
        znear, zfar = 0.1, 100.0
        f = 1.0 / np.tan(fov / 2)
        return np.array([
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (zfar + znear) / (znear - zfar), (2 * zfar * znear) / (znear - zfar)],
            [0, 0, -1, 0]
        ], dtype=np.float32)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.RightButton:
            self.last_mouse_pos = event.pos()

    def mouseMoveEvent(self, event: QMouseEvent):
        if event.buttons() & Qt.RightButton:
            dx = event.x() - self.last_mouse_pos.x()
            dy = event.y() - self.last_mouse_pos.y()
            self.camera_angle += QVector3D(dy * 0.01, dx * 0.01, 0)
            self.last_mouse_pos = event.pos()
            self.update()

    def wheelEvent(self, event: QWheelEvent):
        zoom_change = event.angleDelta().y() * 0.01
        self.camera_pos.setZ(self.camera_pos.z() - zoom_change)  # Zoom entlang der Z-Achse
        self.update()

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.opengl_widget = OpenGLWidget()
        layout.addWidget(self.opengl_widget)
        self.setLayout(layout)
        self.resize(800, 600)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    QSurfaceFormat.setDefaultFormat(fmt)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())