# Auto3D Racetrack Designer

A 3D racetrack design application that allows you to create custom racetracks on procedurally generated terrain.

## Features

- Procedural terrain generation with Perlin noise
- Interactive 3D visualization
- Custom track design with control points
- Track width and banking angle adjustment
- Real-time track mesh generation
- Smooth camera controls

## Installation

1. Clone this repository
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```bash
   python -m auto3d.main
   ```

2. Controls:
   - Left mouse button: Rotate camera
   - Right mouse button: Zoom in/out
   - Left click on terrain: Place track control point
   - Use sliders to adjust track parameters
   - Click "Generate Track" to create the track mesh
   - Click "Clear Track" to start over
   - Click "Regenerate Terrain" to create new terrain

## Requirements

- Python 3.7+
- OpenGL-compatible graphics card
- Required Python packages (see requirements.txt) 