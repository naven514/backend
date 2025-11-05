# Voice Coach Backend API

FastAPI backend for the Voice Coach application.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set environment variables:
- `GEMINI_API_KEY`: Your Google Gemini API key

3. Run the server:
```bash
python app.py
```

Or using uvicorn:
```bash
uvicorn app:app --reload
```

## Deployment on Render

This repository is configured for automatic deployment on Render.

- **Service**: Web Service
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`

## API Endpoints

- `GET /health` - Health check endpoint
- `POST /generate_script` - Generate a presentation script
- `POST /analyze` - Analyze recorded audio

## CORS

The API is configured to accept requests from:
- https://frontend-53528.web.app
- https://frontend-53528.firebaseapp.com

