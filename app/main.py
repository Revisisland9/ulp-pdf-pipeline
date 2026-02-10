from fastapi import FastAPI
from app.api.render import router as render_router

app = FastAPI(
    title="ULP_PDF_PIPELINE",
    version="1.0.0",
)

app.include_router(render_router, prefix="/api/v1")


@app.get("/health")
def health():
    return {"ok": True}
