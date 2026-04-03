# start_all.ps1
# ──────────────────────────────────────────────────────────────────────────────
# Launches the full F1 AI Predictor stack in two separate terminal windows.
#
# Usage (from project root):
#   .\start_all.ps1
#
# Requires:
#   - Python venv at backend\venv\  (create once: python -m venv backend\venv)
#   - Dependencies installed:       pip install -r backend\requirements.txt
# ──────────────────────────────────────────────────────────────────────────────

# 1. Backend — FastAPI on http://localhost:8000
Start-Process powershell -ArgumentList `
    "-NoExit -Command `"Set-Location backend; .\venv\Scripts\Activate.ps1; uvicorn src.api.main:app --reload`""

# 2. Frontend — static file server on http://localhost:3000
Start-Process powershell -ArgumentList `
    "-NoExit -Command `"Set-Location frontend; python -m http.server 3000`""

# 3. Open browser
Start-Sleep -Seconds 2
Start-Process "http://localhost:3000"
