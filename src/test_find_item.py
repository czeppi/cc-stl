import trimesh
from PySide6.QtCore import QPoint

from camera import Camera
from src.itemdetectoratmousepos import ItemDetectorAtMousePos

mesh = trimesh.load("../stl-files/KLP_Lame_Tilted.stl")
camera = Camera(distance=25.0, azimuth=90.0, elevation=-15.0)
view_size = view_width, view_height = 993, 820
#mouse_pos = QPoint(474, 265)
mouse_pos = QPoint(476, 175) #  nearest-face: 12614, found_edge: 30963

aspect_ratio = view_width / view_height
view_matrix = camera.create_view_matrix()
projection_matrix = camera.create_perspective_matrix(aspect_ratio)
mvp_matrix = projection_matrix * view_matrix

item_detector = ItemDetectorAtMousePos(mesh=mesh, mvp_matrix=mvp_matrix, view_size=view_size)
new_mesh_item = item_detector.find_cur_item(mouse_pos)
print(new_mesh_item)
