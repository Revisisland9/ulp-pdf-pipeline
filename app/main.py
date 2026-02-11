from fastapi import FastAPI
from fastapi.responses import Response, JSONResponse
import base64

from app.models import RenderEnvelope
from app.pdf.shipment_confirmation import build_shipment_confirmation_pdf

app = FastAPI(title="ULP_PDF_PIPELINE", version="1.0")

@app.get("/health")
def health():
    return {"ok": True}

# Returns raw PDF bytes (good for browser / curl)
@app.post("/api/v1/render/shipment-confirmation")
def render_shipment_confirmation(payload: RenderEnvelope):
    pdf_bytes = build_shipment_confirmation_pdf(payload.request)

    # Later: use payload.email_to to send email
    # For now: render-only while testing

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": 'inline; filename="shipment_confirmation.pdf"'},
    )

# Returns base64 (best for Apps Script → Drive/email)
@app.post("/api/v1/render/shipment-confirmation/base64")
def render_shipment_confirmation_base64(payload: RenderEnvelope):
    pdf_bytes = build_shipment_confirmation_pdf(payload.request)
    return JSONResponse({
        "filename": "shipment_confirmation.pdf",
        "content_type": "application/pdf",
        "pdf_base64": base64.b64encode(pdf_bytes).decode("utf-8"),
    })
