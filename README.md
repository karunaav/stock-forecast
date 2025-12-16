# RNN Stock Forecast 

A lightweight **FastAPI + Dash** app that trains a **SimpleRNN** model on historical stock closing prices and forecasts the **next-day close**.

## Quick start

### 1) Create & activate a virtual environment
**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS/Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Run the app
```bash
uvicorn app:app --port 8000 --reload
```

Open:
- http://127.0.0.1:8000/

## What you get
- Actual vs Predicted Close plot (test region)
- Current close (latest available)
- Predicted next-day close (from last lookback window)

## Notes
- Data is downloaded via `yfinance` at runtime (so an internet connection is required).
- This is a learning project and **not financial advice**.


