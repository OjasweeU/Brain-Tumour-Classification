# Public Deployment Guide

This repository is now prepared for a public deployment split:

- `Render` for the FastAPI backend
- `Vercel` for the React frontend

## Source Of Truth

`Notebook.ipynb` remains the main model development workflow in this repository. Training, validation, testing, and Grad-CAM experimentation still live there.

Deployment uses the trained artifact only:

- `models/model.keras`
- `app/brain_tumor_ui/inference.py`
- `backend/`
- `frontend/`

## What is already configured

- `render.yaml` defines a Render web service with a health check
- `frontend/vercel.json` adds an SPA rewrite for Vercel
- `requirements.txt` contains the backend runtime dependencies
- `backend/app/main.py` accepts `FRONTEND_ORIGINS` from environment variables for CORS

## What you still need to do manually

Actual public deployment requires your own hosting accounts. This repo cannot create live Render or Vercel projects by itself.

You need to:

1. Push this repository to GitHub.
2. Create a Render web service for the backend.
3. Create a Vercel project for the frontend.
4. Set the correct environment variables in both dashboards.

## Deploy the backend on Render

Render supports `render.yaml` Blueprints and `healthCheckPath` for web services.

Use these settings:

- Runtime: Python
- Build command: `pip install -r requirements.txt`
- Start command: `python backend/run_backend.py`
- Health check path: `/health`

Environment variables to set in Render:

- `BRAIN_TUMOR_MODEL_PATH=models/model.keras`
- `FRONTEND_ORIGINS=https://your-vercel-app-url.vercel.app`
- `BACKEND_PORT=10000`

The backend uses `BACKEND_PORT`, and Render provides the actual listening port through your service config.

## Deploy the frontend on Vercel

Vercel supports Vite projects directly, and Vite reads runtime build variables prefixed with `VITE_`.

Recommended Vercel settings:

- Root directory: `frontend`
- Framework preset: `Vite`
- Build command: `npm run build`
- Output directory: `dist`

Environment variable to set in Vercel:

- `VITE_API_BASE_URL=https://your-render-service.onrender.com`

`frontend/vercel.json` also includes an SPA rewrite so browser refreshes resolve correctly.

## Important note

After you set or change environment variables in Vercel, redeploy the frontend so the new `VITE_API_BASE_URL` is included in the build.

## After deployment

Test these URLs:

- Backend health check: `https://your-render-service.onrender.com/health`
- Frontend app: `https://your-vercel-app-url.vercel.app`

Then upload one MRI image through the frontend and confirm:

- prediction renders
- confidence scores appear
- Grad-CAM image loads

## Sources

- Render health checks and `healthCheckPath`: https://render.com/docs/health-checks
- Render Blueprint fields like `buildCommand` and `startCommand`: https://render.com/docs/blueprint-spec
- Vite on Vercel: https://vercel.com/docs/frameworks/frontend/vite
- Vercel environment variables: https://vercel.com/docs/environment-variables
- Vercel rewrites: https://vercel.com/docs/rewrites/
