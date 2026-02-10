# ULP_PDF_PIPELINE

FastAPI JSON → PDF microservice (Cloud Run friendly)

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8080
