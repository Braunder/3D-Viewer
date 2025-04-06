import sys
import os
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, 
                           QVBoxLayout, QWidget, QFileDialog, QPushButton,
                           QLabel, QStatusBar, QHBoxLayout, QMessageBox, QSizePolicy)
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtCore import Qt, QSize, QThread, pyqtSignal
from OpenGL.GL import *
from OpenGL.GLU import *
import trimesh
from PIL import Image, ImageOps
import pyglet
import json
import open3d as o3d

class ModelLoader(QThread):
    finished = pyqtSignal(object, bool, str)
    error = pyqtSignal(str)

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def find_textures_in_directory(self, model_path):
        """Find texture files in the same directory as the model."""
        model_dir = os.path.dirname(model_path)
        texture_files = {}
        
        # Common texture file patterns
        texture_patterns = [
            '_diffuse', '_color', '_albedo',
            '_basecolor', '_base_color',
            'diffuse', 'color', 'albedo',
            'basecolor', 'base_color',
            '_diff', '_col', '_d',
            '_texture', 'texture',
            ''  # Also try the base filename
        ]
        
        # Common texture extensions
        texture_extensions = ['.png', '.jpg', '.jpeg', '.tga', '.bmp', '.tif', '.tiff']
        
        # Get base name without extension
        base_name = os.path.splitext(os.path.basename(model_path))[0]
        
        print(f"Searching for textures in: {model_dir}")
        print(f"Base model name: {base_name}")
        
        # First, try to find textures with exact matching patterns
        for file in os.listdir(model_dir):
            file_lower = file.lower()
            file_base = os.path.splitext(file_lower)[0]
            
            # Check if file is an image
            if any(file_lower.endswith(ext) for ext in texture_extensions):
                full_path = os.path.join(model_dir, file)
                
                # Check for exact pattern matches
                for pattern in texture_patterns:
                    pattern_to_check = base_name.lower() + pattern
                    if file_base == pattern_to_check:
                        print(f"Found exact match texture: {file}")
                        texture_files[file] = full_path
                        break
        
        # If no exact matches found, try partial matches
        if not texture_files:
            print("No exact matches found, trying partial matches...")
            for file in os.listdir(model_dir):
                file_lower = file.lower()
                
                # Check if file is an image
                if any(file_lower.endswith(ext) for ext in texture_extensions):
                    full_path = os.path.join(model_dir, file)
                    
                    # Check if file contains base name
                    if base_name.lower() in file_lower:
                        print(f"Found partial match texture: {file}")
                        texture_files[file] = full_path
                    # Check for common texture patterns
                    elif any(pattern in file_lower for pattern in texture_patterns):
                        print(f"Found pattern match texture: {file}")
                        texture_files[file] = full_path
        
        if texture_files:
            print(f"Found textures: {list(texture_files.keys())}")
        else:
            print("No textures found")
        
        return texture_files

    def load_with_open3d(self, file_path):
        try:
            print(f"\nLoading model: {file_path}")
            
            # For GLB/GLTF files, try to load with trimesh first to get embedded textures
            if file_path.lower().endswith(('.glb', '.gltf')):
                try:
                    print("Loading GLB/GLTF with trimesh to extract textures...")
                    scene = trimesh.load(file_path)
                    
                    if isinstance(scene, trimesh.Scene):
                        print("Successfully loaded as trimesh Scene")
                        # Get the first mesh and its material
                        if scene.geometry:
                            first_mesh = next(iter(scene.geometry.values()))
                            if hasattr(first_mesh, 'visual') and hasattr(first_mesh.visual, 'material'):
                                material = first_mesh.visual.material
                                if isinstance(material, trimesh.visual.material.PBRMaterial):
                                    print("Found PBR material")
                                    if hasattr(material, 'baseColorTexture') and material.baseColorTexture is not None:
                                        print("Found embedded base color texture")
                                        # Store the texture for later use
                                        self.embedded_texture = material.baseColorTexture
                                        self.has_embedded_texture = True
                except Exception as e:
                    print(f"Error loading with trimesh: {str(e)}")
            
            # Load the mesh using Open3D
            mesh = o3d.io.read_triangle_mesh(file_path)
            
            if not mesh.has_vertices():
                raise Exception("No vertices found in the mesh")
            
            print(f"Mesh loaded: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")
            print(f"Has UV coords: {mesh.has_triangle_uvs()}")
            print(f"Has vertex colors: {mesh.has_vertex_colors()}")
            
            # Convert to numpy arrays
            vertices = np.asarray(mesh.vertices, dtype=np.float32)
            faces = np.asarray(mesh.triangles, dtype=np.uint32)
            
            # Create trimesh mesh
            mesh_data = trimesh.Trimesh(
                vertices=vertices,
                faces=faces
            )
            
            # If the mesh has texture coordinates
            if mesh.has_triangle_uvs():
                print("Processing UV coordinates...")
                uvs = np.asarray(mesh.triangle_uvs, dtype=np.float32)
                
                # Try to use embedded texture first
                if hasattr(self, 'has_embedded_texture') and self.has_embedded_texture:
                    print("Using embedded texture")
                    try:
                        # Create texture visuals with embedded texture
                        material = trimesh.visual.material.SimpleMaterial()
                        material.image = self.embedded_texture
                        mesh_data.visual = trimesh.visual.TextureVisuals(
                            uv=uvs,
                            material=material
                        )
                        print("Embedded texture applied to mesh")
                    except Exception as tex_error:
                        print(f"Error applying embedded texture: {str(tex_error)}")
                else:
                    # Try to find external textures as fallback
                    textures = self.find_textures_in_directory(file_path)
                    if textures:
                        texture_path = next(iter(textures.values()))
                        try:
                            print(f"Loading external texture from: {texture_path}")
                            image = Image.open(texture_path)
                            if image.mode != 'RGBA':
                                image = image.convert('RGBA')
                            print(f"External texture loaded: {image.format} {image.size} {image.mode}")
                            
                            material = trimesh.visual.material.SimpleMaterial()
                            material.image = image
                            mesh_data.visual = trimesh.visual.TextureVisuals(
                                uv=uvs,
                                material=material
                            )
                            print("External texture applied to mesh")
                        except Exception as tex_error:
                            print(f"Error loading external texture: {str(tex_error)}")
                    else:
                        print("No textures found")
                        mesh_data.visual = trimesh.visual.TextureVisuals(uv=uvs)
            
            # If the mesh has vertex colors
            elif mesh.has_vertex_colors():
                print("Using vertex colors...")
                colors = np.asarray(mesh.vertex_colors, dtype=np.float32)
                mesh_data.visual = trimesh.visual.ColorVisuals(vertex_colors=colors)
            
            # Clean up
            if hasattr(self, 'embedded_texture'):
                del self.embedded_texture
                self.has_embedded_texture = False
            
            return mesh_data
            
        except Exception as e:
            print(f"Error loading with Open3D: {str(e)}")
            raise

    def run(self):
        try:
            print(f"Loading model from: {self.file_path}")
            
            # Get file extension
            ext = os.path.splitext(self.file_path)[1].lower()
            
            try:
                # For GLB/GLTF files, try loading with trimesh first
                if ext in ['.glb', '.gltf']:
                    try:
                        loaded = trimesh.load(self.file_path)
                        print("Successfully loaded with trimesh")
                        self.finished.emit(loaded, True, self.file_path)
                        return
                    except Exception as trim_error:
                        print(f"Trimesh loading failed: {str(trim_error)}")
                        print("Falling back to Open3D")
                
                # For other formats or if trimesh failed
                if ext in ['.fbx', '.obj', '.stl', '.ply', '.glb', '.gltf']:
                    try:
                        loaded = self.load_with_open3d(self.file_path)
                        print("Successfully loaded with Open3D")
                    except Exception as o3d_error:
                        print(f"Open3D loading failed: {str(o3d_error)}")
                        print("Falling back to trimesh")
                        loaded = trimesh.load(self.file_path)
                else:
                    loaded = trimesh.load(self.file_path)
                
                if isinstance(loaded, trimesh.Scene):
                    print("Processing trimesh Scene")
                    meshes = []
                    materials = {}
                    
                    for name, mesh in loaded.geometry.items():
                        print(f"Processing mesh: {name}")
                        if isinstance(mesh, trimesh.Trimesh):
                            meshes.append(mesh)
                            # Store material information
                            if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'material'):
                                print(f"Mesh has material: {mesh.visual.material}")
                                if isinstance(mesh.visual.material, trimesh.visual.material.PBRMaterial):
                                    print("Found PBR material")
                                    material = mesh.visual.material
                                    if hasattr(material, 'baseColorTexture') and material.baseColorTexture is not None:
                                        print(f"Material has texture")
                                        # Store the texture image directly
                                        texture_key = f"{name}_baseColor"
                                        materials[texture_key] = material.baseColorTexture
                                        print(f"Stored texture with key: {texture_key}")
                    
                    if not meshes:
                        raise Exception("No valid meshes found in the scene")
                    
                    # Combine all meshes into one
                    if len(meshes) == 1:
                        loaded = meshes[0]
                    else:
                        loaded = trimesh.util.concatenate(meshes)
                
                print("Model loaded successfully")
                self.finished.emit(loaded, True, self.file_path)
                
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                self.error.emit(str(e))
                
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
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Set up light position and properties
        glLightfv(GL_LIGHT0, GL_POSITION, (0, 10, 10, 1))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.2, 0.2, 0.2, 1.0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (1.0, 1.0, 1.0, 1.0))
        glLightfv(GL_LIGHT0, GL_SPECULAR, (1.0, 1.0, 1.0, 1.0))
        
        # Set up material properties
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, (0.2, 0.2, 0.2, 1.0))
        glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, (0.8, 0.8, 0.8, 1.0))
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (1.0, 1.0, 1.0, 1.0))
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 50.0)

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
            
            # Power of 2 check and resize if needed
            width, height = image.size
            if not (self.is_power_of_2(width) and self.is_power_of_2(height)):
                new_width = self.next_power_of_2(width)
                new_height = self.next_power_of_2(height)
                print(f"Resizing texture from {width}x{height} to {new_width}x{new_height}")
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            image_data = image.tobytes()
            
            # Generate texture ID
            texture_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture_id)
            
            # Set texture parameters
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
            
            # Upload texture data and generate mipmaps
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.width, image.height, 
                        0, GL_RGBA, GL_UNSIGNED_BYTE, image_data)
            glGenerateMipmap(GL_TEXTURE_2D)
            
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

    def is_power_of_2(self, n):
        return n != 0 and (n & (n - 1)) == 0

    def next_power_of_2(self, n):
        if n == 0:
            return 1
        n -= 1
        n |= n >> 1
        n |= n >> 2
        n |= n >> 4
        n |= n >> 8
        n |= n >> 16
        return n + 1

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

        # Save current OpenGL state
        glPushAttrib(GL_ALL_ATTRIB_BITS)
        
        try:
            # Enable arrays and bind VBOs
            glEnableClientState(GL_VERTEX_ARRAY)
            glEnableClientState(GL_NORMAL_ARRAY)
            
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_vertices)
            glVertexPointer(3, GL_FLOAT, 0, None)
            
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_normals)
            glNormalPointer(GL_FLOAT, 0, None)
            
            # Handle textures
            if self.vbo_tex_coords is not None and self.materials:
                glEnable(GL_TEXTURE_2D)
                glEnableClientState(GL_TEXTURE_COORD_ARRAY)
                glBindBuffer(GL_ARRAY_BUFFER, self.vbo_tex_coords)
                glTexCoordPointer(2, GL_FLOAT, 0, None)
                
                # Use the first available texture
                texture_key = next(iter(self.materials))
                material_data = self.materials[texture_key]
                if material_data['texture'] is not None:
                    texture_id = self.load_texture(material_data['texture'])
                    if texture_id is not None:
                        print(f"Using texture ID: {texture_id}")
                        glBindTexture(GL_TEXTURE_2D, texture_id)
                        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
            
            # Draw triangles
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.vbo_faces)
            glDrawElements(GL_TRIANGLES, len(self.model.faces) * 3, GL_UNSIGNED_INT, None)
            
        finally:
            # Cleanup state
            glDisableClientState(GL_VERTEX_ARRAY)
            glDisableClientState(GL_NORMAL_ARRAY)
            if self.vbo_tex_coords is not None:
                glDisableClientState(GL_TEXTURE_COORD_ARRAY)
                glDisable(GL_TEXTURE_2D)
                glBindTexture(GL_TEXTURE_2D, 0)
            
            # Restore OpenGL state
            glPopAttrib()

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

    def cleanup_textures(self):
        """Clean up all textures and reset texture-related state"""
        print("Cleaning up textures...")
        try:
            # Delete OpenGL textures
            if hasattr(self, 'textures') and self.textures:
                for texture_id in self.textures.values():
                    try:
                        glDeleteTextures(1, [texture_id])
                        print(f"Deleted texture ID: {texture_id}")
                    except Exception as e:
                        print(f"Error deleting texture {texture_id}: {str(e)}")
                self.textures.clear()
            
            # Clear materials dictionary
            if hasattr(self, 'materials'):
                self.materials.clear()
            
            # Clear embedded texture if any
            if hasattr(self, 'embedded_texture'):
                del self.embedded_texture
                self.has_embedded_texture = False
            
            # Clear texture coordinates
            if hasattr(self, 'tex_coords'):
                self.tex_coords = []
            
            print("Texture cleanup completed")
        except Exception as e:
            print(f"Error during texture cleanup: {str(e)}")

    def load_model(self, file_path):
        """Load a new 3D model"""
        print(f"\nLoading new model: {file_path}")
        
        # Clean up existing textures before loading new model
        self.cleanup_textures()
        
        # Update file path and texture directory
        self.file_path = file_path
        self.texture_dir = os.path.dirname(file_path)
        
        # Start the loader thread
        self.loader = ModelLoader(file_path)
        self.loader.finished.connect(self.on_model_loaded)
        self.loader.error.connect(self.on_model_error)
        self.loader.start()

    def on_model_loaded(self, loaded, success, file_path):
        if success:
            print("Processing loaded model...")
            self.process_loaded_model(loaded)
            self.loader.quit()
            self.loader.wait()
            print("Model processing completed")

    def on_model_error(self, error_msg):
        print(f"Error loading model: {error_msg}")
        QMessageBox.critical(self, "Error", f"Failed to load model: {error_msg}")
        self.loader.quit()
        self.loader.wait()

    def process_loaded_model(self, loaded):
        """Process a newly loaded model"""
        try:
            print("\nProcessing loaded model...")
            
            # Clean up any existing textures before processing new model
            self.cleanup_textures()
            
            if isinstance(loaded, trimesh.Scene):
                print("Processing trimesh Scene")
                # Get all meshes from the scene
                meshes = []
                materials = {}
                
                # Process each mesh in the scene
                for mesh_name, mesh in loaded.geometry.items():
                    print(f"Processing mesh: {mesh_name}")
                    if isinstance(mesh, trimesh.Trimesh):
                        # Store the original visual properties
                        if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'material'):
                            print(f"Mesh has material: {mesh.visual.material}")
                            if isinstance(mesh.visual.material, trimesh.visual.material.PBRMaterial):
                                print("Found PBR material")
                                material = mesh.visual.material
                                if hasattr(material, 'baseColorTexture') and material.baseColorTexture is not None:
                                    print(f"Material has texture")
                                    # Store the texture image and UV coordinates
                                    texture_key = f"{mesh_name}_baseColor"
                                    materials[texture_key] = {
                                        'texture': material.baseColorTexture,
                                        'uv': mesh.visual.uv if hasattr(mesh.visual, 'uv') else None
                                    }
                                    print(f"Stored texture with key: {texture_key}")
                        meshes.append(mesh)
                
                if not meshes:
                    raise Exception("No valid meshes found in the scene")
                
                # If single mesh, use it directly
                if len(meshes) == 1:
                    self.model = meshes[0]
                else:
                    # Combine all meshes while preserving UV coordinates and materials
                    vertices = []
                    faces = []
                    normals = []
                    uvs = []
                    vertex_offset = 0
                    
                    for mesh in meshes:
                        # Add vertices
                        vertices.extend(mesh.vertices.tolist())
                        
                        # Add faces with offset
                        faces.extend((mesh.faces + vertex_offset).tolist())
                        
                        # Add normals
                        if hasattr(mesh, 'vertex_normals'):
                            normals.extend(mesh.vertex_normals.tolist())
                        
                        # Add UV coordinates if available
                        if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'uv'):
                            uvs.extend(mesh.visual.uv.tolist())
                        
                        vertex_offset += len(mesh.vertices)
                    
                    # Create combined mesh
                    vertices = np.array(vertices, dtype=np.float32)
                    faces = np.array(faces, dtype=np.uint32)
                    
                    self.model = trimesh.Trimesh(
                        vertices=vertices,
                        faces=faces,
                        process=False
                    )
                    
                    # Apply UV coordinates if available
                    if uvs:
                        self.model.visual = trimesh.visual.TextureVisuals(uv=np.array(uvs, dtype=np.float32))
                
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
            for texture_key, material_data in self.materials.items():
                print(f"Loading texture for key: {texture_key}")
                if material_data['texture'] is not None:
                    texture_id = self.load_texture(material_data['texture'])
                    if texture_id is not None:
                        print(f"Loaded texture ID {texture_id} for {texture_key}")
                        # Store UV coordinates with the texture
                        if material_data['uv'] is not None:
                            self.textures[texture_key] = {
                                'id': texture_id,
                                'uv': material_data['uv']
                            }
            
            self.update()
            print("Model processing completed successfully")
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
        """Clean up all OpenGL resources"""
        print("Cleaning up OpenGL resources...")
        try:
            # Clean up textures
            self.cleanup_textures()
            
            # Clean up VBOs
            if self.vbo_vertices is not None:
                glDeleteBuffers(1, [self.vbo_vertices])
            if self.vbo_faces is not None:
                glDeleteBuffers(1, [self.vbo_faces])
            if self.vbo_normals is not None:
                glDeleteBuffers(1, [self.vbo_normals])
            if self.vbo_tex_coords is not None:
                glDeleteBuffers(1, [self.vbo_tex_coords])
            
            print("OpenGL cleanup completed")
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")

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
        
        # Create main vertical layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        
        # Create OpenGL widget first
        self.viewer = ModelViewer()
        # Set size policy to make viewer expand in both directions
        self.viewer.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )
        
        # Create top toolbar layout
        toolbar_layout = QHBoxLayout()
        toolbar_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        toolbar_layout.setSpacing(5)
        
        # Create load button
        load_button = QPushButton("Load Model")
        load_button.setFixedWidth(100)
        load_button.clicked.connect(self.load_model)
        toolbar_layout.addWidget(load_button)
        
        # Create texture directory button
        texture_button = QPushButton("Load Textures")
        texture_button.setFixedWidth(100)
        texture_button.clicked.connect(self.viewer.load_textures_from_directory)
        toolbar_layout.addWidget(texture_button)
        
        # Info label
        self.info_label = QLabel("No model loaded")
        toolbar_layout.addWidget(self.info_label)
        
        # Add stretch to push everything to the left
        toolbar_layout.addStretch()
        
        # Add layouts and widgets to main layout
        main_layout.addLayout(toolbar_layout)
        main_layout.addWidget(self.viewer)
        
        # Set window size and title
        self.setWindowTitle("3D Model Viewer")
        self.resize(1024, 768)  # Default window size
        
        # Remove margin between toolbar and viewer
        main_layout.setSpacing(0)

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
