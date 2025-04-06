# 3D Model Viewer

A Python-based 3D model viewer application built with PyQt6 and OpenGL. This viewer supports loading and displaying various 3D model formats with texture support.

![3D Model](https://i.imgur.com/jguCVyw.png)

## Features

- Support for multiple 3D model formats (OBJ, FBX, GLTF, GLB, STL, PLY)
- Texture support for models
- Interactive model manipulation:
  - Rotation (left mouse button)
  - Zoom (mouse wheel)
- Modern UI with PyQt6
- OpenGL-based rendering

## Known Limitations

- **FBX Format Issues**:
  - Textures may not load correctly with FBX models
  - Some complex FBX models may not display properly

- **Model Size and Parts**:
  - Complex models with multiple parts may not display correctly
  - Very large models may have performance issues
  - Some models may require manual scaling adjustment

- **Best Performance**:
  - Works best with simple models
  - GLB and GLTF formats are recommended for best compatibility
  - Models with standard UV mapping work best

## Requirements

- Python 3.x
- PyQt6
- OpenGL
- trimesh
- numpy
- open3d
- PIL (Python Imaging Library)

## Installation

1. Clone the repository
2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
python main.py
```

2. Use the "Load Model" button to open a 3D model file
3. Use the "Load Textures" button to load additional textures if needed
4. Interact with the model:
   - Left mouse button: Rotate
   - Mouse wheel: Zoom in/out

## Supported File Formats

- OBJ
- FBX (with limitations)
- GLTF
- GLB
- STL
- PLY

## License

[Add your license information here]

## Acknowledgments

- PyQt6 for the GUI framework
- OpenGL for 3D rendering capabilities
- Trimesh for 3D model processing
- The Python community for various supporting libraries 
