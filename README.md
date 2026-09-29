
# Exchange Rate Forecasting — Model Serving Guide

ARIMA(1,1,1) model trained on daily exchange rate data.  
Two serving approaches are implemented: **MLflow** and **Docker + FastAPI**.

---

## Architecture Overview

### How the Files Connect

```
exchange_rate.csv
      │
      ▼
time_series_forecast.py
  ├── Loads & preprocesses data
  ├── Trains ARIMA(1,1,1) model
  ├── Evaluates on test set (MAE, RMSE, MAPE)
  ├── Saves ──────────────────────► model.pkl
  └── Logs to MLflow ─────────────► mlruns/
                                      └── experiment/
                                            ├── params
                                            ├── metrics
                                            └── artifacts/
                                                  └── model (PyFunc)
```

---

### Way 1 — MLflow Flow

```
mlruns/artifacts/model
      │
      ▼
mlflow models serve          ← reads the PyFunc wrapper + model.pkl from mlruns
      │
      ▼
ARIMAForecastModel.predict() ← custom class intercepts steps, calls model.forecast()
      │
      ▼
uvicorn server @ port 5000
      │
      ▼
POST /invocations             ← Postman sends {"dataframe_split": {"columns": ["steps"], "data": [[30]]}}
      │
      ▼
JSON response with forecast array
```

---

### Way 2 — Docker Flow

```
requirements.txt ──┐
model.pkl ─────────┤
app.py ────────────┤──► Dockerfile packages all 3 ──► Docker Image (tsa-arima-model)
                   │                                        │
                   └────────────────────────────────────────┘
                                                            │
                                                     docker run
                                                            │
                                                            ▼
                                                  app.py starts FastAPI
                                                  loads model.pkl into memory
                                                            │
                                                            ▼
                                               uvicorn server @ port 8000
                                                            │
                                                            ▼
                                              POST /forecast  {"steps": 30}
                                                            │
                                                            ▼
                                              JSON response with forecast array
```

---

### Role of Each File

| File | What it does |
|---|---|
| `exchange_rate.csv` | Raw input data — daily exchange rates |
| `time_series_forecast.py` | Trains ARIMA, evaluates, saves `model.pkl`, logs to MLflow |
| `model.pkl` | Serialized trained ARIMA model — used by both serving methods |
| `app.py` | FastAPI app — loads `model.pkl` at startup, exposes `/forecast` endpoint |
| `Dockerfile` | Picks `requirements.txt` + `model.pkl` + `app.py`, packages into a container image. `WORKDIR /app` creates an isolated working directory inside the container so all files are organized under `/app` and uvicorn can locate `app.py` at startup |
| `requirements.txt` | Dependency list installed inside the Docker container |
| `mlruns/` | MLflow tracking store — stores params, metrics, and the PyFunc model artifact |

---

## Prerequisites

```bash
# Activate the virtual environment first
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
# Train the model and log to MLflow (generates model.pkl + mlruns/)
python time_series_forecast.py
```

---

## Way 1 — MLflow Model Serving

### Start the server
```bash
mlflow models serve \
  -m runs:/3ec3bb1321f7480292cb4770f430aee9/model \
  -p 5000 \
  --env-manager local
```
Server runs at: `http://127.0.0.1:5000`

### Test in Postman

| Field | Value |
|---|---|
| Method | `POST` |
| URL | `http://127.0.0.1:5000/invocations` |
| Headers → Key | `Content-Type` |
| Headers → Value | `application/json` |
| Body | `raw` → `JSON` |


**Raw JSON body:**
```json
{
    "dataframe_split": {
        "columns": ["steps"],
        "data": [[30]]
    }
}
```

> The model is wrapped in a **custom PyFunc class** (`ARIMAForecastModel`) that intercepts the input, extracts the `steps` integer, and calls `model.forecast(steps)` internally. This eliminates the need for `start/end` date ranges required by the default MLflow statsmodels flavor.

<img width="500" height="500" alt="Screenshot 2026-09-29 at 12 10 08 AM" src="https://github.com/user-attachments/assets/8bd00ad7-3720-4a26-a0a7-e1d7b576c929" />


---

## Way 2 — Docker + FastAPI Serving

### Build the image
```bash
docker build -t tsa-arima-model .
```

### Run the container
```bash
docker run -d -p 8000:8000 --name tsa-server tsa-arima-model
```
Server runs at: `http://localhost:8000`

### Test in Postman

#### Health Check
| Field | Value |
|---|---|
| Method | `GET` |
| URL | `http://localhost:8000/` |

Expected response:
```json
{"status": "ok", "model": "ARIMA(1,1,1)"}
```

#### Forecast
| Field | Value |
|---|---|
| Method | `POST` |
| URL | `http://localhost:8000/forecast` |
| Headers → Key | `Content-Type` |
| Headers → Value | `application/json` |
| Body | `raw` → `JSON` |

**Raw JSON body** (just pass the number of days you want):
```json
{"steps": 30}
```

Expected response:
```json
{
    "steps": 30,
    "forecast": [0.7208, 0.7208, ...]
}
```

<img width="500" height="500" alt="Screenshot 2026-09-29 at 12 18 09 AM" src="https://github.com/user-attachments/assets/39999a1b-fac1-4885-b0d7-67dfe8cf327f" />


### Container management
```bash
docker ps                    # Check running container
docker logs tsa-server       # View server logs
docker stop tsa-server       # Stop the container
docker start tsa-server      # Start it again
docker rm -f tsa-server      # Remove the container
```

---

## Comparison

| | MLflow Serve | Docker + FastAPI |
|---|---|---|
| **Endpoint** | `/invocations` | `/forecast` |
| **Input format** | `dataframe_split` with `steps` | `{"steps": N}` |
| **Best for** | Quick local experimentation | Production / deployment |
| **Port** | `5000` | `8000` |

---

## File Structure

```
.
├── time_series_forecast.py   # Training + MLflow logging
├── app.py                    # FastAPI serving app
├── Dockerfile                # Docker build definition
├── requirements.txt          # Python dependencies
├── model.pkl                 # Serialized ARIMA model
└── mlruns/                   # MLflow experiment tracking data
```
