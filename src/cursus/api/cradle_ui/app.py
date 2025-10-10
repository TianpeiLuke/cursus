"""
FastAPI application for Cradle Data Load Config UI

This module provides the main FastAPI application with all routes and middleware.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import logging
import os
import re

from .api.routes import router
from .schemas.response_schemas import ErrorResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Security functions to prevent path traversal attacks
def validate_file_path(file_path: str, allowed_dir: str) -> str:
    """
    Validate that a file path is within the allowed directory.

    Args:
        file_path: The file path to validate
        allowed_dir: The allowed base directory

    Returns:
        str: Validated absolute file path

    Raises:
        HTTPException: If path is outside allowed directory
    """
    try:
        # Convert to absolute paths
        abs_file_path = os.path.abspath(file_path)
        abs_allowed_dir = os.path.abspath(allowed_dir)

        # Check if file path is within allowed directory
        if not abs_file_path.startswith(abs_allowed_dir + os.sep):
            raise HTTPException(
                status_code=403, detail="Access denied: Path outside allowed directory"
            )

        return abs_file_path
    except Exception as e:
        logger.error(f"Path validation failed: {e}")
        raise HTTPException(status_code=400, detail="Invalid file path")


# Create FastAPI application
app = FastAPI(
    title="Cradle Data Load Config UI API",
    description="REST API for the Cradle Data Load Configuration wizard interface",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Include API routes
app.include_router(router)


# Global exception handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions with consistent error format."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(error=exc.detail, code=str(exc.status_code)).model_dump(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error", detail=str(exc), code="500"
        ).model_dump(),
    )


# Root endpoint - serve the main UI
@app.get("/")
async def root():
    """Serve the main UI page."""
    from fastapi.responses import FileResponse

    static_dir = os.path.join(os.path.dirname(__file__), "static")
    index_file = os.path.join(static_dir, "index.html")

    if os.path.exists(index_file):
        # Validate that index.html is within the static directory
        validated_path = validate_file_path(index_file, static_dir)
        return FileResponse(validated_path)
    else:
        # Fallback to API info if HTML file not found
        return {
            "name": "Cradle Data Load Config UI API",
            "version": "1.0.0",
            "description": "REST API for the Cradle Data Load Configuration wizard interface",
            "docs_url": "/docs",
            "health_url": "/api/cradle-ui/health",
            "error": "UI file not found - check static/index.html",
        }


# API info endpoint
@app.get("/api/info")
async def api_info():
    """API information endpoint."""
    return {
        "name": "Cradle Data Load Config UI API",
        "version": "1.0.0",
        "description": "REST API for the Cradle Data Load Configuration wizard interface",
        "docs_url": "/docs",
        "health_url": "/api/cradle-ui/health",
    }


# Health check endpoint
@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "cradle-ui-api"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
