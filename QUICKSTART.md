# Medical Cancer Expert System — Quick Start Guide

## Option A: Standard Mode (no Docker)

### 1. Backend setup
```bash
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac
pip install -r requirements.txt
cp .env.example .env
# Edit .env: add your GROQ_API_KEY
python main_v2.py
# → http://localhost:8000
```

### 2. Frontend setup
```bash
cd frontend
npm install
npm run dev
# → http://localhost:3000
```

### 3. Login
- URL: http://localhost:3000
- Username: `admin`  Password: `admin`

---

## Option B: Docker Compose (all services)

```bash
cp backend/.env.example backend/.env
# Edit backend/.env with your GROQ_API_KEY

docker-compose -f docker/docker-compose.yml up --build
```

Services started:
| Service    | URL                        |
|------------|----------------------------|
| Frontend   | http://localhost:3000      |
| Backend    | http://localhost:8000      |
| API Docs   | http://localhost:8000/docs |
| Grafana    | http://localhost:3001      |
| Prometheus | http://localhost:9090      |

---

## Train the Cancer Detection Model

```bash
cd backend
venv\Scripts\python.exe train_model.py
# Trains ResNet50 for 30 epochs with class weights + cosine LR
# Output: cancer_model.pth, label_classes.json, metrics.json
```

## Run Evaluation (Confusion Matrices)

```bash
venv\Scripts\python.exe evaluate_model.py
# Output: confusion_matrix.png, confusion_matrix_binary.png
#         confusion_matrix_brain.png, confusion_matrix_lung.png
#         confusion_matrix_skin.png, metrics.json
```

## Train with Differential Privacy

```bash
venv\Scripts\python.exe -m ai.dp_training
# Output: cancer_model_dp.pth, dp_metrics.json
```

## Run Federated Learning (3 hospitals)

```bash
# Terminal 1 — FL Server
venv\Scripts\python.exe -m ai.federated.fl_server

# Terminal 2 — Hospital A
venv\Scripts\python.exe -m ai.federated.fl_client --hospital A

# Terminal 3 — Hospital B
venv\Scripts\python.exe -m ai.federated.fl_client --hospital B
```

---

## New Features

### 🔐 Encrypted AI (FHE)
- Click the 🔐 button in the chat input area
- Select mode: Standard / Concrete ML / OpenFHE
- Upload a medical image
- View privacy workflow and timing breakdown

### 📊 Benchmark Dashboard
- Click "Benchmark" in the sidebar
- Upload any medical image to run all 3 modes
- View accuracy vs latency vs privacy comparison table

### 👨‍⚕️ Doctor Dashboard
- Click "Doctor Dashboard" in the sidebar
- View patient scan timelines and disease progression
- Add clinical notes per patient

### 🔬 Grad-CAM Heatmaps
```bash
POST /inference/gradcam
# Upload image → get base64 heatmap overlay
# Shows which regions drove the prediction
```

---

## API Quick Reference

```
GET  /health                    System health check
GET  /docs                      Interactive API documentation
POST /auth/login                Login → JWT token
POST /inference/analyze?mode=X  Inference (standard/concrete_ml/openfhe)
GET  /inference/modes           Available modes + status
POST /inference/gradcam         Grad-CAM heatmap
POST /inference/benchmark/quick Compare all 3 modes
GET  /benchmark/compare         Latest results per mode
GET  /doctor/dashboard/stats    Doctor overview stats
GET  /doctor/patients           Patient list
GET  /metrics                   Prometheus metrics
```

---

## Verify Everything Works

```bash
# Backend health
curl http://localhost:8000/health

# Available inference modes
curl http://localhost:8000/inference/modes

# API documentation
open http://localhost:8000/docs
```
