# Deployment Guide - Shipment Delay Analyzer

This guide explains how to deploy the Shipment Delay Analyzer system, which consists of:
1. **Interactive Dashboard (Streamlit)**: A user-friendly web interface (`app.py`).
2. **REST API (FastAPI)**: An API endpoint (`api.py`) for programmatic integrations.

---

## ☁️ Deploying to Streamlit Cloud (GitHub)

The `ModuleNotFoundError` you encountered occurs because Streamlit Cloud needs to know which external Python packages to install. 

We have added a `requirements.txt` file at the root of the repository listing all dependencies (`streamlit`, `pandas`, `numpy`, `joblib`, `requests`, `xgboost`, `fastapi`, `uvicorn`, `pydantic`, `scikit-learn`).

### Steps to Deploy:
1. **Commit and Push Changes**:
   Make sure you commit and push the new `requirements.txt` file to your GitHub repository:
   ```bash
   git add requirements.txt
   git commit -m "Add requirements.txt to fix streamlit cloud deployment"
   git push origin main
   ```
2. **Reboot the App**:
   - Go to your Streamlit Cloud dashboard: [share.streamlit.io](https://share.streamlit.io/).
   - Locate your app.
   - Click the three dots `...` next to the app and click **Settings**.
   - Scroll down and click **Reboot App**. Streamlit will now read `requirements.txt` and install all necessary dependencies (including `joblib` and `xgboost`).

---

## 🐳 Containerized Deployment (Docker & Docker Compose)

You can containerize and run both the API and Streamlit Dashboard locally or on a cloud virtual machine (e.g., AWS EC2, GCP VM, DigitalOcean Droplet) using Docker.

### Prerequisites
* [Docker](https://www.docker.com/products/docker-desktop/) installed.

### Steps:
1. **Build and Start Containers**:
   From your project root directory, run:
   ```bash
   docker compose up --build
   ```
2. **Access the Services**:
   * **Streamlit Dashboard**: Open `http://localhost:8501` in your browser.
   * **FastAPI REST API Docs**: Open `http://localhost:8000/docs` in your browser.
   * **FastAPI Health Check**: Access `http://localhost:8000/` in your browser or terminal.

3. **Stop the Containers**:
   ```bash
   docker compose down
   ```

---

## 🐍 Manual Local Run (Python Virtual Environment)

If you prefer to run the services directly in your terminal without Docker:

### Steps:
1. **Create and Activate a Virtual Environment**:
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the REST API**:
   ```bash
   uvicorn api:app --reload --host 0.0.0.0 --port 8000
   ```

4. **Run the Streamlit Dashboard**:
   ```bash
   streamlit run app.py --server.port=8501
   ```

---

## 📡 Testing the REST API Endpoints

Once the API is running (either locally, via Docker, or deployed to a server), you can test it with the following requests:

### 1. Health Check
* **Method**: `GET`
* **URL**: `http://localhost:8000/`
* **Response**:
  ```json
  {
    "status": "online",
    "model_loaded": true,
    "features_loaded": 49
  }
  ```

### 2. Single Shipment Delay Prediction
* **Method**: `POST`
* **URL**: `http://localhost:8000/predict`
* **Headers**: `Content-Type: application/json`
* **Body** (using variables matching `sample_shipment.json`):
  ```json
  {
    "shipping_mode": "Sea",
    "shipment_type": "Import",
    "priority": "Standard",
    "weight_kg": 5000.0,
    "distance_km": 1250.0,
    "weather_condition": "Clear",
    "insurance": false
  }
  ```
* **Command Line Test (PowerShell)**:
  ```powershell
  Invoke-RestMethod -Uri http://localhost:8000/predict -Method Post -ContentType "application/json" -InFile sample_shipment.json
  ```
* **Command Line Test (cURL)**:
  ```bash
  curl -X POST "http://localhost:8000/predict?threshold=0.5" \
       -H "Content-Type: application/json" \
       -d @sample_shipment.json
  ```
