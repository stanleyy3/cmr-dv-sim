"""
Auto3D - A 3D Racetrack Design and Simulation Package
"""

import open3d as o3d
from packaging import version
import sys

MIN_OPEN3D_VERSION = "0.17.0"

def check_dependencies():
    """Check if all required dependencies are available and compatible."""
    try:
        # Check Open3D version
        current_version = o3d.__version__
        if version.parse(current_version) < version.parse(MIN_OPEN3D_VERSION):
            raise ImportError(
                f"Open3D version {MIN_OPEN3D_VERSION} or higher is required. "
                f"Found version {current_version}"
            )
            
        # Check if GUI is available
        if not hasattr(o3d, 'visualization') or not hasattr(o3d.visualization, 'gui'):
            raise ImportError("Open3D GUI module is not available. Please install Open3D with GUI support.")
            
        # Don't initialize GUI here, just check if the modules are available
        if not hasattr(o3d.visualization.gui, 'Application'):
            raise ImportError("Open3D GUI Application module is not available.")
            
        return True
            
    except Exception as e:
        print(f"Error checking dependencies: {e}", file=sys.stderr)
        return False

# Make dependency check available but don't run automatically
is_compatible = check_dependencies() 