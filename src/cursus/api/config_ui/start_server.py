#!/usr/bin/env python3
"""
Simple server startup script for Config UI that fixes import issues.

This script follows the same pattern as Cradle UI to handle imports correctly.
"""

import sys
from pathlib import Path

# Add project root to path (same pattern as Cradle UI)
project_root = str(Path(__file__).parent.parent.parent.parent)
sys.path.insert(0, project_root)

def start_server(host="127.0.0.1", port=8003):
    """Start the Config UI server with proper imports."""
    
    print("🚀 Starting Universal Config UI Server...")
    print(f"📍 Host: {host}")
    print(f"🔌 Port: {port}")
    print(f"📁 Project Root: {project_root}")
    print()
    print("✨ Enhanced Features:")
    print("  • Universal configuration support")
    print("  • DAG-driven pipeline configuration")
    print("  • Real-time validation and error handling")
    print("  • Save All Merged functionality")
    print("  • Jupyter widget integration")
    print()
    print("🌐 Access the UI at:")
    print(f"  • Web Interface: http://{host}:{port}/config-ui")
    print(f"  • API Documentation: http://{host}:{port}/docs")
    print(f"  • Health Check: http://{host}:{port}/health")
    print()
    
    try:
        # Handle both relative and absolute imports using centralized path setup
        try:
            # Try relative imports first (when run as module)
            from .web.api import create_config_ui_app
        except ImportError:
            # Fallback: Set up cursus path and use absolute imports
            import sys
            from pathlib import Path
            
            # Add the core directory to path for import_utils
            current_dir = Path(__file__).parent
            core_dir = current_dir / 'core'
            if str(core_dir) not in sys.path:
                sys.path.insert(0, str(core_dir))
            
            from core.import_utils import ensure_cursus_path
            ensure_cursus_path()
            
            from cursus.api.config_ui.web.api import create_config_ui_app
        
        import uvicorn
        
        # Create the FastAPI app
        app = create_config_ui_app()
        
        # Run the server
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info",
            access_log=True
        )
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print()
        print("💡 Make sure you're running from the correct directory:")
        print("   cd /path/to/cursus")
        print("   python src/cursus/api/config_ui/start_server.py")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Server Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Start Universal Config UI Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8003, help="Port to bind to")
    
    args = parser.parse_args()
    start_server(args.host, args.port)
