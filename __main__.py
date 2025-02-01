"""Main entry point for Auto3D"""

import sys
import traceback
from . import is_compatible
from .main import main

def run_application():
    """Run the Auto3D application with proper error handling."""
    try:
        # Check dependencies first
        if not is_compatible:
            print("Error: System compatibility check failed. Please check the requirements.",
                  file=sys.stderr)
            sys.exit(1)
            
        # Initialize and run the application
        print("Initializing Auto3D...")
        main()
        
    except KeyboardInterrupt:
        print("\nApplication terminated by user.")
        sys.exit(0)
        
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        print("\nDetailed error information:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
        
if __name__ == "__main__":
    run_application() 