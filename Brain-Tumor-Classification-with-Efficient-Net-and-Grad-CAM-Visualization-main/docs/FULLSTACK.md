# React + FastAPI Setup

This project now has a full-stack deployment layer that leaves the notebook-based training workflow untouched.

## Source Of Truth

`Notebook.ipynb` is still the core ML pipeline in this repository. It is responsible for training, testing, validation, and evaluation.

The backend and frontend are deployment layers built around the trained model artifact. They do not replace the notebook and they do not define the training logic.

## Structure

- `Notebook.ipynb`: main ML workflow and validation pipeline
- `backend/app/main.py`: FastAPI inference API
- `backend/requirements.txt`: backend-only dependencies
- `backend/run_backend.py`: local backend launcher
- `frontend/`: React + Vite frontend
- `app/brain_tumor_ui/inference.py`: shared inference logic
- `models/model.keras`: trained model artifact used by the backend

## Local run

### 1. Start the backend

Install backend dependencies:

```bash
pip install -r requirements.txt
```

Run the API from the project root:

```bash
python backend/run_backend.py
```

The backend will start on `http://localhost:8000`.

Health check:

```bash
curl http://localhost:8000/health
```

### 2. Start the React frontend

From the project root:

```bash
cd frontend
npm install
npm run dev
```

The frontend will start on `http://localhost:5173`.

If PowerShell blocks `npm`, use:

```bash
cd frontend
npm.cmd install
npm.cmd run dev
```

## Environment

If your backend is not on the default local URL, create `frontend/.env`:

```bash
VITE_API_BASE_URL=http://localhost:8000
```

## CV angle

This version is stronger than a notebook-only project because it demonstrates:

- ML inference service design
- frontend-backend integration
- explainable AI output in a user-facing product
- separation between training code and deployment code

A good one-line description for this setup is:

`Trained and validated an EfficientNet-based brain tumor MRI classifier in Jupyter, then deployed the trained model through a FastAPI backend and React frontend for interactive inference and Grad-CAM visualization.`
