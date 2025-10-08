#!/usr/bin/env python3
"""
Startup script for Config UI Server

This script properly runs the Config UI API server as a module,
ensuring all relative imports work correctly for package portability.

Usage:
    python run_server.py [--host HOST] [--port PORT] [--reload]

Examples:
    python run_server.py
    python run_server.py --host 0.0.0.0 --port 8003
    python run_server.py --reload  # For development
"""

import argparse
import sys
import uvicorn
from pathlib import Path

def main():
    """Main entry point for the Config UI server."""
    parser = argparse.ArgumentParser(description="Run Config UI Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8003, help="Port to bind to (default: 8003)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"], 
                       help="Log level (default: info)")
    
    args = parser.parse_args()
    
    print("🚀 Starting Enhanced Config UI Server...")
    print(f"📍 Host: {args.host}")
    print(f"🔌 Port: {args.port}")
    print(f"🔄 Reload: {args.reload}")
    print(f"📊 Log Level: {args.log_level}")
    print()
    print("✨ Enhanced Features:")
    print("  • Request deduplication and caching")
    print("  • Debounced field validation")
    print("  • Global state management")
    print("  • Enhanced error handling")
    print("  • Robust UI patterns from Cradle UI")
    print()
    print("🌐 Access the UI at:")
    print(f"  • Web Interface: http://{args.host}:{args.port}/config-ui")
    print(f"  • API Documentation: http://{args.host}:{args.port}/docs")
    print(f"  • Health Check: http://{args.host}:{args.port}/health")
    print()
    
    try:
        # Try relative import first (when run as module)
        try:
            from .web.api import create_config_ui_app
        except ImportError:
            # Fallback to absolute import (when run as script)
            # Add current directory to path to enable absolute imports
            current_dir = Path(__file__).parent
            if str(current_dir) not in sys.path:
                sys.path.insert(0, str(current_dir))
            
            from web.api import create_config_ui_app
        
        # Create the FastAPI app
        app = create_config_ui_app()
        
        # Run the server
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level=args.log_level,
            access_log=True
        )
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print()
        print("💡 Troubleshooting:")
        print("   Method 1 - Run as module from project root:")
        print("   cd /path/to/cursus")
        print("   python -m src.cursus.api.config_ui.run_server")
        print()
        print("   Method 2 - Run as module from src directory:")
        print("   cd src")
        print("   python -m cursus.api.config_ui.run_server")
        print()
        print("   Method 3 - Run directly from config_ui directory:")
        print("   cd src/cursus/api/config_ui")
        print("   python run_server.py")
        print()
        print(f"   Current working directory: {Path.cwd()}")
        print(f"   Script location: {Path(__file__).parent}")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Server Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
