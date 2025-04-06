import sys
import os
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, 
                           QVBoxLayout, QWidget, QFileDialog, QPushButton,
                           QLabel, QStatusBar, QHBoxLayout)
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtCore import Qt, QSize, QThread, pyqtSignal
from OpenGL.GL import *
from OpenGL.GLU import *
import trimesh

class ModelLoader(QThread):
    finished = pyqtSignal(object, bool)
    error = pyqtSignal(str)

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def run(self):
        try:
            loaded = trimesh.load(self.file_path)
            self.finished.emit(loaded, True)
        except Exception as e:
            self.error.emit(str(e))

class ModelViewer(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = None
        self.rotation = [0, 0, 0]
        self.scale = 1.0
        self.translation = [0, 0, -5]
        self.last_pos = None
        self.file_path = None
        self.vertices = []
        self.faces = []
        self.normals = []
        self.using_trimesh = False
        self.using_custom = False
        self.vbo_vertices = None
        self.vbo_faces = None
        self.vbo_normals = None
        self.setMinimumSize(400, 400)

    def minimumSizeHint(self):
        return QSize(400, 400)

    def initializeGL(self):
        glClearColor(0.2, 0.2, 0.2, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        
        # Set up light position
        glLight(GL_LIGHT0, GL_POSITION, (0, 0, 1, 0))

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, w / h, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def create_vbo(self, data, target):
        vbo = glGenBuffers(1)
        glBindBuffer(target, vbo)
        glBufferData(target, data.nbytes, data, GL_STATIC_DRAW)
        return vbo

    def setup_vbo(self):
        if self.using_trimesh and self.model is not None:
            # Create VBOs for trimesh model
            vertices = np.array(self.model.vertices, dtype=np.float32)
            faces = np.array(self.model.faces, dtype=np.uint32)
            normals = np.array(self.model.vertex_normals, dtype=np.float32)
            
            self.vbo_vertices = self.create_vbo(vertices, GL_ARRAY_BUFFER)
            self.vbo_faces = self.create_vbo(faces, GL_ELEMENT_ARRAY_BUFFER)
            self.vbo_normals = self.create_vbo(normals, GL_ARRAY_BUFFER)
        elif self.using_custom and self.vertices and self.faces:
            # Create VBOs for custom model
            vertices = np.array(self.vertices, dtype=np.float32)
            faces = np.array(self.faces, dtype=np.uint32)
            normals = np.array(self.normals, dtype=np.float32)
            
            self.vbo_vertices = self.create_vbo(vertices, GL_ARRAY_BUFFER)
            self.vbo_faces = self.create_vbo(faces, GL_ELEMENT_ARRAY_BUFFER)
            self.vbo_normals = self.create_vbo(normals, GL_ARRAY_BUFFER)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # Apply transformations
        glTranslatef(*self.translation)
        glRotatef(self.rotation[0], 1, 0, 0)
        glRotatef(self.rotation[1], 0, 1, 0)
        glRotatef(self.rotation[2], 0, 0, 1)
        glScalef(self.scale, self.scale, self.scale)

        if self.using_trimesh and self.model is not None:
            self.draw_trimesh_model()
        elif self.using_custom and self.vertices and self.faces:
            self.draw_custom_model()

    def draw_trimesh_model(self):
        if self.model is None or self.vbo_vertices is None:
            return

        glColor3f(0.8, 0.8, 0.8)  # Light gray
        
        # Enable vertex and normal arrays
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        
        # Bind VBOs
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_vertices)
        glVertexPointer(3, GL_FLOAT, 0, None)
        
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_normals)
        glNormalPointer(GL_FLOAT, 0, None)
        
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.vbo_faces)
        
        # Draw triangles
        glDrawElements(GL_TRIANGLES, len(self.model.faces) * 3, GL_UNSIGNED_INT, None)
        
        # Disable arrays
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)

    def draw_custom_model(self):
        if not self.vertices or not self.faces or self.vbo_vertices is None:
            return

        glColor3f(0.8, 0.8, 0.8)  # Light gray
        
        # Enable vertex and normal arrays
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        
        # Bind VBOs
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_vertices)
        glVertexPointer(3, GL_FLOAT, 0, None)
        
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_normals)
        glNormalPointer(GL_FLOAT, 0, None)
        
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.vbo_faces)
        
        # Draw triangles
        glDrawElements(GL_TRIANGLES, len(self.faces) * 3, GL_UNSIGNED_INT, None)
        
        # Disable arrays
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)

    def load_model(self, file_path):
        self.file_path = file_path
        self.loader = ModelLoader(file_path)
        self.loader.finished.connect(self.on_model_loaded)
        self.loader.error.connect(self.on_model_error)
        self.loader.start()

    def on_model_loaded(self, loaded, success):
        if success:
            self.process_loaded_model(loaded)
            self.loader.quit()
            self.loader.wait()

    def on_model_error(self, error_msg):
        print(f"Error loading model: {error_msg}")
        self.loader.quit()
        self.loader.wait()

    def process_loaded_model(self, loaded):
        try:
            if isinstance(loaded, trimesh.Scene):
                # Get all meshes from the scene
                meshes = []
                for mesh_name, mesh in loaded.geometry.items():
                    if isinstance(mesh, trimesh.Trimesh):
                        meshes.append(mesh)
                
                if not meshes:
                    raise Exception("No valid meshes found in the scene")
                
                # Combine all meshes into one
                if len(meshes) == 1:
                    self.model = meshes[0]
                else:
                    self.model = trimesh.util.concatenate(meshes)
                
                self.using_trimesh = True
                self.using_custom = False
            else:
                self.model = loaded
                self.using_trimesh = True
                self.using_custom = False
            
            # Center the model
            self.model.vertices -= self.model.center_mass
            
            # Scale model to fit in view
            max_dimension = np.max(self.model.extents)
            scale_factor = 2.0 / max_dimension if max_dimension > 0 else 1.0
            self.model.vertices *= scale_factor
            
            # Reset view parameters
            self.rotation = [0, 0, 0]
            self.scale = 1.0
            self.translation = [0, 0, -5]
            
            # Setup VBOs
            self.setup_vbo()
            
            self.update()
        except Exception as e:
            print(f"Error processing model: {e}")
            self.using_trimesh = False
            self.using_custom = False
            self.model = None

    def extract_scene_data(self, scene):
        self.vertices = []
        self.faces = []
        self.normals = []
        self.using_trimesh = False
        self.using_custom = True
        
        vertex_offset = 0
        for mesh_name, mesh in scene.geometry.items():
            # Add vertices
            self.vertices.extend(mesh.vertices.tolist())
            
            # Add faces with offset
            for face in mesh.faces:
                self.faces.append([vertex_offset + i for i in face])
            
            # Add normals
            self.normals.extend(mesh.vertex_normals.tolist())
            
            vertex_offset += len(mesh.vertices)
        
        # Center and scale
        if self.vertices:
            vertices_array = np.array(self.vertices)
            center = np.mean(vertices_array, axis=0)
            self.vertices = (vertices_array - center).tolist()
            
            # Scale
            min_vals = np.min(vertices_array, axis=0)
            max_vals = np.max(vertices_array, axis=0)
            max_range = np.max(max_vals - min_vals)
            scale_factor = 2.0 / max_range if max_range > 0 else 1.0
            
            self.vertices = (np.array(self.vertices) * scale_factor).tolist()
        
        # Reset view parameters
        self.rotation = [0, 0, 0]
        self.scale = 1.0
        self.translation = [0, 0, -5]
        
        # Setup VBOs
        self.setup_vbo()
        
        self.update()
        return True

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_pos = event.position()
        event.accept()

    def mouseMoveEvent(self, event):
        if self.last_pos is None:
            event.accept()
            return
            
        if event.buttons() & Qt.MouseButton.LeftButton:
            current_pos = event.position()
            dx = current_pos.x() - self.last_pos.x()
            dy = current_pos.y() - self.last_pos.y()
            
            # Rotate around Y axis for horizontal movement
            self.rotation[1] += dx * 0.5
            # Rotate around X axis for vertical movement
            self.rotation[0] += dy * 0.5
            
            self.update()
        
        self.last_pos = event.position()
        event.accept()

    def wheelEvent(self, event):
        # Get the angle delta from the wheel event
        delta = event.angleDelta().y()
        
        # Calculate zoom factor
        zoom_factor = 1.1 if delta > 0 else 0.9
        
        # Apply zoom
        self.scale *= zoom_factor
        
        # Limit zoom range
        self.scale = max(0.1, min(self.scale, 10.0))
        
        self.update()
        event.accept()

    def cleanup(self):
        # Cleanup VBOs
        if self.vbo_vertices is not None:
            glDeleteBuffers(1, [self.vbo_vertices])
        if self.vbo_faces is not None:
            glDeleteBuffers(1, [self.vbo_faces])
        if self.vbo_normals is not None:
            glDeleteBuffers(1, [self.vbo_normals])

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Model Viewer")
        self.setup_ui()
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Create OpenGL widget
        self.viewer = ModelViewer()
        main_layout.addWidget(self.viewer)

        # Button layout
        button_layout = QHBoxLayout()
        
        # Create load button
        load_button = QPushButton("Load Model")
        load_button.clicked.connect(self.load_model)
        button_layout.addWidget(load_button)
        
        # Info label
        self.info_label = QLabel("No model loaded")
        button_layout.addWidget(self.info_label)
        
        main_layout.addLayout(button_layout)
        
        self.setMinimumSize(800, 600)

    def load_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open 3D Model",
            "",
            "3D Models (*.obj *.fbx *.gltf *.glb)"
        )
        if file_path:
            self.info_label.setText("Loading model...")
            self.status_bar.showMessage("Loading...")
            self.viewer.load_model(file_path)

    def closeEvent(self, event):
        self.viewer.cleanup()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
