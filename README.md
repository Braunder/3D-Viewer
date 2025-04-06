# 3D Model Viewer

A versatile 3D model viewer built with Python, PyQt6, and OpenGL. This application allows you to view various 3D model formats with support for textures and materials.

## Features

- Support for multiple 3D model formats:
  - OBJ (.obj)
  - FBX (.fbx)
  - GLTF/GLB (.gltf, .glb)
- Texture support:
  - Embedded textures (GLTF/GLB)
  - External texture files
  - PBR materials
- Interactive controls:
  - Rotate model with left mouse button
  - Zoom with mouse wheel
- Modern UI with PyQt6
- Real-time OpenGL rendering

## Requirements

- Python 3.8 or higher
- PyQt6
- OpenGL
- Trimesh
- NumPy
- Pillow (PIL)
- PyGLet

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/3d-model-viewer.git
cd 3d-model-viewer
```

2. Create and activate a virtual environment (recommended):
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
python main.py
```

2. Use the interface:
   - Click "Load Model" to open a 3D model file
   - Click "Load Textures" to select a directory with external texture files (if needed)
   - Use the mouse to interact with the model:
     - Left mouse button: Rotate the model
     - Mouse wheel: Zoom in/out

## Supported File Formats

### Models
- OBJ (.obj) - Wavefront OBJ format
- FBX (.fbx) - Autodesk FBX format
- GLTF/GLB (.gltf, .glb) - Khronos Group's GL Transmission Format

### Textures
- PNG (.png)
- JPEG (.jpg, .jpeg)
- BMP (.bmp)
- TGA (.tga)

## Controls

- **Left Mouse Button**: Hold and drag to rotate the model
- **Mouse Wheel**: Scroll to zoom in/out
- **Load Model**: Opens a file dialog to select a 3D model
- **Load Textures**: Opens a directory dialog to select external textures

## Troubleshooting

### Common Issues

1. **Textures not loading:**
   - Check if texture files are in the same directory as the model
   - Ensure texture file names match the model's references
   - Try using the "Load Textures" button to specify texture directory

2. **Model appears blank or incorrect:**
   - Verify the file format is supported
   - Check if the model has valid geometry
   - Ensure all required textures are available

3. **Performance issues:**
   - Large models may take longer to load
   - Complex textures might affect rendering performance

## Development

The application is built using:
- PyQt6 for the user interface
- OpenGL for 3D rendering
- Trimesh for 3D model loading and processing
- PIL (Python Imaging Library) for texture handling

### Project Structure

```
3d-model-viewer/
├── main.py          # Main application file
├── requirements.txt # Python dependencies
└── README.md       # This file
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PyQt6 for the GUI framework
- OpenGL for 3D rendering capabilities
- Trimesh for 3D model processing
- The Python community for various supporting libraries 