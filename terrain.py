import numpy as np
from noise import snoise2
from scipy.ndimage import gaussian_filter

class TerrainGenerator:
    def __init__(self, width=100, height=100, scale=50.0):
        """
        Initialize the terrain generator.
        
        Args:
            width (int): Number of points in x direction
            height (int): Number of points in y direction
            scale (float): Scale factor for the terrain
        """
        self.width = width
        self.height = height
        self.scale = scale
        self.terrain = np.zeros((height, width))
        
    def add_slope(self, x_gradient=0.1, y_gradient=0.1):
        """Add an overall slope to the terrain."""
        x = np.linspace(0, 1, self.width)
        y = np.linspace(0, 1, self.height)
        X, Y = np.meshgrid(x, y)
        
        slope = X * x_gradient + Y * y_gradient
        self.terrain += slope * self.scale
        
    def add_perlin_noise(self, octaves=6, persistence=0.5, lacunarity=2.0):
        """Add Perlin noise to the terrain."""
        for y in range(self.height):
            for x in range(self.width):
                nx = x/self.width - 0.5
                ny = y/self.height - 0.5
                self.terrain[y][x] += snoise2(nx, ny, 
                                            octaves=octaves,
                                            persistence=persistence,
                                            lacunarity=lacunarity) * self.scale
                
    def smooth_terrain(self, sigma=1.0):
        """Smooth the terrain using Gaussian filtering."""
        self.terrain = gaussian_filter(self.terrain, sigma=sigma)
        
    def generate(self, x_gradient=0.1, y_gradient=0.1, noise_scale=1.0):
        """Generate complete terrain with slope and noise."""
        self.terrain = np.zeros((self.height, self.width))
        self.add_slope(x_gradient, y_gradient)
        self.add_perlin_noise()
        self.smooth_terrain()
        return self.terrain
        
    def get_height_at_point(self, x, y):
        """Get interpolated height at any point."""
        x_idx = int(x * (self.width - 1))
        y_idx = int(y * (self.height - 1))
        return self.terrain[y_idx, x_idx] 