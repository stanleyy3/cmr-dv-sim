import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

class Visualizer:
    def __init__(self, width=1024, height=768):
        """Initialize the Open3D visualizer with GUI."""
        # Initialize application
        if not gui.Application.instance.initialized():
            gui.Application.instance.initialize()
            
        # Create window with a scene widget
        self.window = gui.Application.instance.create_window(
            "Auto3D Track Designer", width, height)
        
        # Create scene widget
        self.scene = gui.SceneWidget()
        self.scene.enable_scene_caching(True)  # Improve performance
        
        # Enable mouse controls for camera (use rotate mode for more intuitive control)
        self.scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)
        self.scene.set_view_controls_enabled(True)
        
        self.window.add_child(self.scene)
        
        # Setup UI panel
        em = self.window.theme.font_size
        panel_width = 20 * em
        
        # Create panel with fixed width
        panel = gui.Vert(0, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em))
        panel.preferred_width = panel_width
        
        # Track controls
        panel.add_child(gui.Label("Track Controls"))
        
        # Track width slider
        self.track_width = gui.Slider(gui.Slider.DOUBLE)
        self.track_width.set_limits(5.0, 20.0)
        self.track_width.double_value = 10.0
        self.track_width.set_on_value_changed(self._on_parameter_changed)
        panel.add_child(gui.Label("Track Width"))
        panel.add_child(self.track_width)
        
        # Banking angle slider
        self.banking_angle = gui.Slider(gui.Slider.DOUBLE)
        self.banking_angle.set_limits(-30.0, 30.0)
        self.banking_angle.double_value = 0.0
        self.banking_angle.set_on_value_changed(self._on_parameter_changed)
        panel.add_child(gui.Label("Banking Angle"))
        panel.add_child(self.banking_angle)
        
        # Point counter
        self.point_label = gui.Label("Points placed: 0")
        panel.add_child(self.point_label)
        
        # Buttons
        button_layout = gui.Horiz(0.4 * em)
        self.clear_button = gui.Button("Clear Track")
        self.generate_button = gui.Button("Generate Track")
        button_layout.add_child(self.clear_button)
        button_layout.add_child(self.generate_button)
        panel.add_child(button_layout)
        
        # Terrain controls
        panel.add_child(gui.Label("\nTerrain Controls"))
        
        self.terrain_scale = gui.Slider(gui.Slider.DOUBLE)
        self.terrain_scale.set_limits(10.0, 100.0)
        self.terrain_scale.double_value = 50.0
        self.terrain_scale.set_on_value_changed(self._on_parameter_changed)
        panel.add_child(gui.Label("Terrain Scale"))
        panel.add_child(self.terrain_scale)
        
        self.regen_terrain_button = gui.Button("Regenerate Terrain")
        panel.add_child(self.regen_terrain_button)
        
        # Instructions
        panel.add_child(gui.Label("\nControls:"))
        panel.add_child(gui.Label("- Left click: Place track point"))
        panel.add_child(gui.Label("- Left click + drag: Rotate camera"))
        panel.add_child(gui.Label("- Right click + drag: Pan"))
        panel.add_child(gui.Label("- Mouse wheel: Zoom"))
        
        # Add panel to window
        self.window.add_child(panel)
        
        # Setup scene
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.scene.scene.set_background([0.8, 0.8, 0.8, 1.0])
        
        # Setup lighting
        self._setup_lighting()
        
        # Store geometries and callbacks
        self.terrain_mesh = None
        self.track_mesh = None
        self.point_callback = None
        self.clear_callback = None
        self.generate_callback = None
        self.regen_terrain_callback = None
        
        # Store materials
        self.terrain_material = None
        self.track_material = None
        
        # Setup mouse interaction
        self.scene.set_on_mouse(self._mouse_event)
        
        # Store window dimensions
        self.width = width
        self.height = height
        
        # Set up window layout and handle resizing
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_resize(self._on_resize)
        self._on_layout(self.window.content_rect)
        
    def _setup_lighting(self):
        """Setup scene lighting."""
        scene = self.scene.scene.scene
        scene.enable_sun_light(True)
        scene.set_sun_light(
            [-1, -1, -2],  # direction
            [1, 1, 1],     # color
            75000)         # intensity
        scene.enable_indirect_light(True)
        scene.set_indirect_light_intensity(10000)
        scene.enable_ambient_occlusion(True)
        
    def _on_layout(self, layout_context):
        """Handle window layout."""
        # Get window dimensions
        content_rect = self.window.content_rect
        panel_width = self.window.theme.font_size * 20
        
        # Position scene widget
        scene_rect = gui.Rect(0, 0, 
                            content_rect.width - panel_width,
                            content_rect.height)
        self.scene.frame = scene_rect
        
        # Position control panel
        panel_rect = gui.Rect(content_rect.width - panel_width, 0,
                           panel_width, content_rect.height)
        self.window.get_child(1).frame = panel_rect
        
        # Update camera bounds if terrain exists
        if self.terrain_mesh is not None:
            bounds = self.terrain_mesh.get_axis_aligned_bounding_box()
            self._update_camera(bounds)
            
    def _on_resize(self, width, height):
        """Handle window resize events."""
        self.width = width
        self.height = height
        
    def _on_parameter_changed(self, value):
        """Handle parameter changes."""
        if self.generate_callback:
            self.generate_callback()
            
    def _update_camera(self, bounds):
        """Update camera to fit geometry."""
        if bounds is None:
            return
            
        # Calculate center and size of bounds
        center = bounds.get_center()
        size = bounds.get_max_extent()
        
        # Set up camera view
        fov = 60
        distance = size * 2.0  # Ensure full geometry is visible
        
        # Update camera parameters
        self.scene.setup_camera(fov, bounds, center)
        self.scene.look_at(center, center + [0, 0, distance], [0, 1, 0])
        
    def _create_terrain_material(self):
        """Create material for terrain."""
        mat = rendering.MaterialRecord()
        mat.shader = "defaultLit"
        mat.base_color = [0.7, 0.7, 0.7, 1.0]
        mat.roughness = 0.7
        mat.metallic = 0.0
        mat.base_reflectance = 0.1
        return mat
        
    def _create_track_material(self):
        """Create material for track."""
        mat = rendering.MaterialRecord()
        mat.shader = "defaultLit"
        mat.base_color = [0.3, 0.3, 0.3, 1.0]
        mat.roughness = 0.9
        mat.metallic = 0.1
        mat.base_reflectance = 0.2
        return mat
        
    def set_terrain_mesh(self, vertices, faces):
        """Set the terrain mesh data."""
        if vertices is None or faces is None:
            print("Invalid terrain mesh data")
            return
            
        try:
            # Clean up old mesh
            if self.terrain_mesh is not None:
                self.scene.scene.clear_geometry()
                self.terrain_mesh = None
            
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.triangles = o3d.utility.Vector3iVector(faces)
            mesh.compute_vertex_normals()
            mesh.compute_triangle_normals()
            
            self.terrain_mesh = mesh
            
            # Create new material
            self.terrain_material = self._create_terrain_material()
            
            self.scene.scene.add_geometry("terrain", self.terrain_mesh, 
                                        self.terrain_material)
            
            # Update camera to fit new geometry
            bounds = self.terrain_mesh.get_axis_aligned_bounding_box()
            self._update_camera(bounds)
        except Exception as e:
            print(f"Error setting terrain mesh: {e}")
        
    def set_track_mesh(self, vertices, faces):
        """Set the track mesh data."""
        if vertices is None or faces is None:
            if self.track_mesh is not None:
                self.scene.scene.remove_geometry("track")
                self.track_mesh = None
            return
            
        try:
            # Clean up old mesh
            if self.track_mesh is not None:
                self.scene.scene.remove_geometry("track")
                self.track_mesh = None
            
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.triangles = o3d.utility.Vector3iVector(faces)
            mesh.compute_vertex_normals()
            mesh.compute_triangle_normals()
            
            self.track_mesh = mesh
            
            # Create new material
            self.track_material = self._create_track_material()
            
            self.scene.scene.add_geometry("track", self.track_mesh,
                                        self.track_material)
        except Exception as e:
            print(f"Error setting track mesh: {e}")
        
    def set_callbacks(self, point_cb, clear_cb, generate_cb, regen_terrain_cb):
        """Set all UI callbacks."""
        self.point_callback = point_cb
        self.clear_callback = clear_cb
        self.generate_callback = generate_cb
        self.regen_terrain_callback = regen_terrain_cb
        
        self.clear_button.set_on_clicked(self.clear_callback)
        self.generate_button.set_on_clicked(self.generate_callback)
        self.regen_terrain_button.set_on_clicked(self.regen_terrain_callback)
        
    def update_point_count(self, count):
        """Update the point counter label."""
        self.point_label.text = f"Points placed: {count}"
        
    def _mouse_event(self, event):
        """Handle mouse events for point placement."""
        # Check if mouse is over UI panel
        if event.x > self.scene.frame.width:
            return gui.Widget.EventCallbackResult.IGNORED
            
        # Let the scene widget handle camera controls first
        if self.scene.handle_mouse(event) == gui.Widget.EventCallbackResult.HANDLED:
            return gui.Widget.EventCallbackResult.HANDLED
            
        # Handle point placement
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and \
           event.is_button_down(gui.MouseButton.LEFT):
            # Get ray from camera
            x = event.x - self.scene.frame.x
            y = event.y - self.scene.frame.y
            ray = self.scene.scene.camera.unproject(
                x, y, self.scene.frame.width, self.scene.frame.height)
            
            # Cast ray against terrain
            result = self.scene.scene.ray_cast(ray[0], ray[1], rendering.Scene.MESH_TYPE)
            
            if result.hit_geometry and result.geometry_name == "terrain":
                point = result.point
                if self.point_callback:
                    self.point_callback(point[0], point[1])
                    
        return gui.Widget.EventCallbackResult.HANDLED
        
    def get_track_params(self):
        """Get current track parameters."""
        return {
            'width': self.track_width.double_value,
            'banking': self.banking_angle.double_value,
            'terrain_scale': self.terrain_scale.double_value
        }
        
    def run(self):
        """Start the application."""
        gui.Application.instance.run()
        
    def close(self):
        """Close the application."""
        try:
            # Clean up resources
            self.scene.scene.clear_geometry()
            self.terrain_mesh = None
            self.track_mesh = None
            self.terrain_material = None
            self.track_material = None
            gui.Application.instance.quit()
        except Exception as e:
            print(f"Error during cleanup: {e}") 