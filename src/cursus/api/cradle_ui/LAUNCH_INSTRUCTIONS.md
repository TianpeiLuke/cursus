# How to Launch the Cradle Data Load Config UI

## Quick Start

### Option 1: Using Python Module (Recommended)
```bash
# From the cursus project root directory
python -m cursus.api.cradle_ui.app
```

### Option 2: Direct Python Execution
```bash
# Navigate to the cradle_ui directory
cd src/cursus/api/cradle_ui
python app.py
```

### Option 3: Using uvicorn directly
```bash
# From the cursus project root directory
uvicorn cursus.api.cradle_ui.app:app --host 0.0.0.0 --port 8000 --reload
```

## Access the UI

Once the server starts, you'll see output like:
```
INFO:     Started server process [XXXXX]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

**Open your web browser and navigate to:**
- **Main UI**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs (Swagger UI)
- **Alternative API Docs**: http://localhost:8000/redoc

## Troubleshooting

### Port Already in Use
If you see "Address already in use" error:
```bash
# Kill existing process on port 8000
lsof -ti:8000 | xargs kill -9

# Or use a different port
uvicorn cursus.api.cradle_ui.app:app --host 0.0.0.0 --port 8001
```

### Dependencies Missing
If you get import errors, install dependencies:
```bash
pip install fastapi uvicorn python-multipart
```

## Using the UI

1. **Step 1: Data Sources** - Configure project settings, time range, and data sources
2. **Step 2: Transform** - Set up SQL transformation and job splitting options  
3. **Step 3: Output** - Configure output schema and format settings
4. **Step 4: Cradle Job** - Set cluster and job execution parameters
5. **Final Step** - Select job type and generate the CradleDataLoadConfig

The UI will guide you through each step and generate a complete, valid CradleDataLoadConfig object at the end.

## API Endpoints

The backend provides these REST API endpoints:
- `GET /` - Serve the main UI
- `GET /api/health` - Health check
- `POST /api/build-config` - Build CradleDataLoadConfig from form data
- `GET /api/field-schema` - Get dynamic form schema
- `POST /api/validate-config` - Validate configuration
- `GET /api/configs` - List saved configurations
- `POST /api/configs` - Save configuration
- `GET /api/configs/{config_id}` - Get specific configuration

## Development Mode

For development with auto-reload:
```bash
uvicorn cursus.api.cradle_ui.app:app --reload --host 0.0.0.0 --port 8000
