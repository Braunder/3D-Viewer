import sys
import os
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, 
                           QVBoxLayout, QWidget, QFileDialog, QPushButton,
                           QLabel, QStatusBar, QHBoxLayout, QMessageBox)
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtCore import Qt, QSize, QThread, pyqtSignal
from OpenGL.GL import *
from OpenGL.GLU import *
import trimesh
from PIL import Image, ImageOps
import pyglet
import json

class ModelLoader(QThread):
    finished = pyqtSignal(object, bool, str)
    error = pyqtSignal(str)

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def run(self):
        try:
            print(f"Loading model from: {self.file_path}")
            loaded = trimesh.load(self.file_path)
            self.finished.emit(loaded, True, self.file_path)
        except Exception as e:
            print(f"Error in ModelLoader: {str(e)}")
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
        self.texture_dir = None
        self.vertices = []
        self.faces = []
        self.normals = []
        self.tex_coords = []
        self.textures = {}
        self.materials = {}
        self.using_trimesh = False
        self.using_custom = False
        self.vbo_vertices = None
        self.vbo_faces = None
        self.vbo_normals = None
        self.vbo_tex_coords = None
        self.setMinimumSize(400, 400)

    def minimumSizeHint(self):
        return QSize(400, 400)

    def initializeGL(self):
        glClearColor(0.2, 0.2, 0.2, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_TEXTURE_2D)
        
        # Set up light position
        glLight(GL_LIGHT0, GL_POSITION, (0, 0, 1, 0))
        
        # Set up material properties
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, (0.2, 0.2, 0.2, 1.0))
        glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, (0.8, 0.8, 0.8, 1.0))
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (0.0, 0.0, 0.0, 1.0))
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 0.0)

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

    def load_texture(self, image_path):
        try:
            # Handle both file paths and PIL Image objects
            if isinstance(image_path, str):
                print(f"Attempting to load texture from path: {image_path}")
                if image_path in self.textures:
                    print(f"Texture already loaded, returning cached texture ID: {self.textures[image_path]}")
                    return self.textures[image_path]
                
                # Try to load from texture directory if path is relative
                if self.texture_dir and not os.path.isabs(image_path):
                    possible_paths = [
                        os.path.join(self.texture_dir, image_path),
                        os.path.join(self.texture_dir, os.path.basename(image_path)),
                        # Try common texture file extensions
                        os.path.join(self.texture_dir, os.path.splitext(image_path)[0] + '.png'),
                        os.path.join(self.texture_dir, os.path.splitext(image_path)[0] + '.jpg'),
                        os.path.join(self.texture_dir, os.path.splitext(image_path)[0] + '.jpeg')
                    ]
                    
                    for path in possible_paths:
                        print(f"Trying texture path: {path}")
                        if os.path.exists(path):
                            print(f"Found texture at: {path}")
                            image_path = path
                            break
                    else:
                        print(f"No texture found in any of the attempted paths")
                        return None
                
                image = Image.open(image_path)
                print(f"Successfully loaded image: {image.format} {image.size} {image.mode}")
            else:
                print("Processing PIL Image object directly")
                image = image_path
            
            # Convert image to RGBA if needed
            if image.mode != 'RGBA':
                print(f"Converting image from {image.mode} to RGBA")
                image = image.convert('RGBA')
            
            # Flip the image vertically (OpenGL uses bottom-left origin)
            image = ImageOps.flip(image)
            image_data = image.tobytes()
            
            # Generate texture ID
            texture_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture_id)
            
            # Set texture parameters
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
            
            # Upload texture data
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.width, image.height, 
                        0, GL_RGBA, GL_UNSIGNED_BYTE, image_data)
            
            print(f"Successfully created texture ID: {texture_id}")
            
            # Store texture ID using a unique identifier
            texture_key = str(id(image_path)) if isinstance(image_path, str) else str(id(image_path))
            self.textures[texture_key] = texture_id
            return texture_id
            
        except Exception as e:
            print(f"Error loading texture: {e}")
            import traceback
            traceback.print_exc()
            return None

    def setup_vbo(self):
        if self.using_trimesh and self.model is not None:
            # Create VBOs for trimesh model
            vertices = np.array(self.model.vertices, dtype=np.float32)
            faces = np.array(self.model.faces, dtype=np.uint32)
            normals = np.array(self.model.vertex_normals, dtype=np.float32)
            
            self.vbo_vertices = self.create_vbo(vertices, GL_ARRAY_BUFFER)
            self.vbo_faces = self.create_vbo(faces, GL_ELEMENT_ARRAY_BUFFER)
            self.vbo_normals = self.create_vbo(normals, GL_ARRAY_BUFFER)
            
            # Add texture coordinates if available
            if hasattr(self.model, 'visual') and hasattr(self.model.visual, 'uv'):
                tex_coords = np.array(self.model.visual.uv, dtype=np.float32)
                self.vbo_tex_coords = self.create_vbo(tex_coords, GL_ARRAY_BUFFER)
        elif self.using_custom and self.vertices and self.faces:
            # Create VBOs for custom model
            vertices = np.array(self.vertices, dtype=np.float32)
            faces = np.array(self.faces, dtype=np.uint32)
            normals = np.array(self.normals, dtype=np.float32)
            
            self.vbo_vertices = self.create_vbo(vertices, GL_ARRAY_BUFFER)
            self.vbo_faces = self.create_vbo(faces, GL_ELEMENT_ARRAY_BUFFER)
            self.vbo_normals = self.create_vbo(normals, GL_ARRAY_BUFFER)
            
            if self.tex_coords:
                tex_coords = np.array(self.tex_coords, dtype=np.float32)
                self.vbo_tex_coords = self.create_vbo(tex_coords, GL_ARRAY_BUFFER)

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

        # Enable vertex and normal arrays
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        
        # Bind VBOs
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_vertices)
        glVertexPointer(3, GL_FLOAT, 0, None)
        
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_normals)
        glNormalPointer(GL_FLOAT, 0, None)
        
        # Enable and bind texture coordinates if available
        if self.vbo_tex_coords is not None:
            print("Binding texture coordinates")
            glEnableClientState(GL_TEXTURE_COORD_ARRAY)
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_tex_coords)
            glTexCoordPointer(2, GL_FLOAT, 0, None)
            
            # Apply texture if available
            if self.materials:
                # Use the first available texture
                texture_key = next(iter(self.materials))
                texture = self.materials[texture_key]
                print(f"Using texture: {texture_key}")
                texture_id = self.load_texture(texture)
                if texture_id is not None:
                    print(f"Binding texture ID: {texture_id}")
                    glBindTexture(GL_TEXTURE_2D, texture_id)
        
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.vbo_faces)
        
        # Draw triangles
        glDrawElements(GL_TRIANGLES, len(self.model.faces) * 3, GL_UNSIGNED_INT, None)
        
        # Disable arrays
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)
        if self.vbo_tex_coords is not None:
            glDisableClientState(GL_TEXTURE_COORD_ARRAY)
            glBindTexture(GL_TEXTURE_2D, 0)

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
        self.texture_dir = os.path.dirname(file_path)
        self.loader = ModelLoader(file_path)
        self.loader.finished.connect(self.on_model_loaded)
        self.loader.error.connect(self.on_model_error)
        self.loader.start()

    def on_model_loaded(self, loaded, success, file_path):
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
                print("Processing trimesh Scene")
                # Get all meshes from the scene
                meshes = []
                materials = {}
                
                for mesh_name, mesh in loaded.geometry.items():
                    print(f"Processing mesh: {mesh_name}")
                    if isinstance(mesh, trimesh.Trimesh):
                        meshes.append(mesh)
                        # Store material information
                        if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'material'):
                            print(f"Mesh has material: {mesh.visual.material}")
                            if isinstance(mesh.visual.material, trimesh.visual.material.PBRMaterial):
                                print("Found PBR material")
                                material = mesh.visual.material
                                if hasattr(material, 'baseColorTexture') and material.baseColorTexture is not None:
                                    print(f"Material has texture: {material.baseColorTexture}")
                                    # Store the texture image directly
                                    texture_key = f"{mesh_name}_baseColor"
                                    materials[texture_key] = material.baseColorTexture
                                    print(f"Stored texture with key: {texture_key}")
                
                if not meshes:
                    raise Exception("No valid meshes found in the scene")
                
                # Combine all meshes into one
                if len(meshes) == 1:
                    self.model = meshes[0]
                else:
                    self.model = trimesh.util.concatenate(meshes)
                
                # Store the materials dictionary
                self.materials = materials
                
                self.using_trimesh = True
                self.using_custom = False
            else:
                print("Processing single trimesh")
                self.model = loaded
                self.using_trimesh = True
                self.using_custom = False
            
            # Print model information
            print(f"Model loaded: vertices={len(self.model.vertices)}, faces={len(self.model.faces)}")
            if hasattr(self.model, 'visual'):
                print(f"Model has visual properties: {self.model.visual}")
            
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
            
            # Load textures from materials
            for texture_key, texture in self.materials.items():
                print(f"Loading texture for key: {texture_key}")
                self.load_texture(texture)
            
            self.update()
        except Exception as e:
            print(f"Error processing model: {e}")
            import traceback
            traceback.print_exc()
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
        if self.vbo_tex_coords is not None:
            glDeleteBuffers(1, [self.vbo_tex_coords])
            
        # Cleanup textures
        for texture_id in self.textures.values():
            glDeleteTextures(1, [texture_id])
        self.textures.clear()

    def load_textures_from_directory(self):
        if not self.texture_dir:
            QMessageBox.warning(self, "Warning", "No model loaded. Please load a model first.")
            return

        texture_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Texture Directory",
            self.texture_dir
        )
        
        if texture_dir:
            print(f"Selected texture directory: {texture_dir}")
            self.texture_dir = texture_dir
            
            # Clear existing textures
            for texture_id in self.textures.values():
                glDeleteTextures(1, [texture_id])
            self.textures.clear()
            
            # Print available texture files in directory
            print("Available texture files in directory:")
            texture_files = []
            for file in os.listdir(texture_dir):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tga')):
                    print(f" - {file}")
                    texture_files.append(os.path.join(texture_dir, file))
            
            # Try to load external textures if no embedded textures
            if not self.materials and texture_files:
                print("No embedded textures found, trying to load external textures")
                for texture_file in texture_files:
                    if "basecolor" in texture_file.lower():
                        print(f"Loading external texture: {texture_file}")
                        self.materials[os.path.basename(texture_file)] = texture_file
            
            self.update()

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
        
        # Create texture directory button
        texture_button = QPushButton("Load Textures")
        texture_button.clicked.connect(self.viewer.load_textures_from_directory)
        button_layout.addWidget(texture_button)
        
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
            "3D Models (*.obj *.fbx *.gltf *.glb);;All Files (*.*)"
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
