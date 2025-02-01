import numpy as np
from .terrain import TerrainGenerator
from .track import TrackGenerator
from .visualizer import Visualizer

class Auto3DApp:
    def __init__(self):
        # Initialize components
        self.terrain_gen = TerrainGenerator(width=100, height=100)
        self.track_gen = TrackGenerator()
        self.visualizer = Visualizer()
        
        # Generate initial terrain
        self.terrain = self.terrain_gen.generate()
        
        # Setup visualization and callbacks
        self.setup_visualization()
        
    def setup_visualization(self):
        """Setup the visualization and callbacks."""
        try:
            # Convert terrain to mesh
            vertices, faces = self.terrain_to_mesh()
            self.visualizer.set_terrain_mesh(vertices, faces)
            
            # Set up callbacks
            self.visualizer.set_callbacks(
                point_cb=self.add_track_point,
                clear_cb=self.clear_track,
                generate_cb=self.generate_track,
                regen_terrain_cb=self.regenerate_terrain
            )
        except Exception as e:
            print(f"Error setting up visualization: {e}")
            raise
        
    def add_track_point(self, x, y):
        """Add a track point at the clicked location."""
        try:
            # Validate coordinates
            if not (0 <= x < self.terrain_gen.width and 0 <= y < self.terrain_gen.height):
                print(f"Point ({x}, {y}) is outside terrain bounds")
                return
                
            params = self.visualizer.get_track_params()
            self.track_gen.add_point(x, y, params['width'], params['banking'])
            self.visualizer.update_point_count(len(self.track_gen.control_points))
            
            # Auto-generate track if we have enough points
            if len(self.track_gen.control_points) >= 2:
                self.generate_track()
        except Exception as e:
            print(f"Error adding track point: {e}")
        
    def terrain_to_mesh(self):
        """Convert terrain height map to 3D mesh."""
        if self.terrain is None:
            raise ValueError("Terrain not generated")
            
        vertices = []
        faces = []
        
        try:
            # Generate vertices
            for y in range(self.terrain_gen.height):
                for x in range(self.terrain_gen.width):
                    vertices.append([x, y, self.terrain[y, x]])
                    
            # Generate faces
            for y in range(self.terrain_gen.height - 1):
                for x in range(self.terrain_gen.width - 1):
                    v0 = y * self.terrain_gen.width + x
                    v1 = v0 + 1
                    v2 = (y + 1) * self.terrain_gen.width + x
                    v3 = v2 + 1
                    
                    faces.extend([
                        [v0, v2, v1],
                        [v1, v2, v3]
                    ])
                    
            return np.array(vertices), np.array(faces)
        except Exception as e:
            print(f"Error generating terrain mesh: {e}")
            raise
        
    def clear_track(self):
        """Clear the current track."""
        try:
            self.track_gen.clear_points()
            self.visualizer.set_track_mesh(None, None)
            self.visualizer.update_point_count(0)
        except Exception as e:
            print(f"Error clearing track: {e}")
        
    def generate_track(self):
        """Generate the track mesh from control points."""
        try:
            if len(self.track_gen.control_points) < 2:
                return
                
            # Generate spline points
            spline_points = self.track_gen.generate_spline()
            if spline_points is None:
                print("Failed to generate spline points")
                return
                
            # Get terrain heights at spline points
            heights = []
            for x, y in spline_points:
                # Clamp coordinates to terrain bounds
                x = np.clip(x, 0, self.terrain_gen.width - 1)
                y = np.clip(y, 0, self.terrain_gen.height - 1)
                heights.append(self.terrain_gen.get_height_at_point(x, y))
            
            heights = np.array(heights)
            
            # Generate track mesh
            vertices, faces = self.track_gen.generate_track_mesh(spline_points, heights)
            if vertices is not None and faces is not None:
                self.visualizer.set_track_mesh(vertices, faces)
        except Exception as e:
            print(f"Error generating track: {e}")
        
    def regenerate_terrain(self):
        """Regenerate the terrain with current parameters."""
        try:
            params = self.visualizer.get_track_params()
            self.terrain_gen.scale = params['terrain_scale']
            self.terrain = self.terrain_gen.generate()
            
            vertices, faces = self.terrain_to_mesh()
            self.visualizer.set_terrain_mesh(vertices, faces)
            
            # Clear track when terrain changes
            self.clear_track()
        except Exception as e:
            print(f"Error regenerating terrain: {e}")
        
    def run(self):
        """Start the application."""
        try:
            self.visualizer.run()
        except Exception as e:
            print(f"Error running application: {e}")
            raise
        
    def close(self):
        """Close the application."""
        try:
            self.visualizer.close()
        except Exception as e:
            print(f"Error closing application: {e}")

def main():
    try:
        app = Auto3DApp()
        app.run()
    except Exception as e:
        print(f"Fatal error: {e}")
    finally:
        try:
            app.close()
        except:
            pass  # Ignore errors during emergency shutdown

if __name__ == "__main__":
    main() 