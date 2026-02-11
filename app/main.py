from fastapi import FastAPI, Body
from fastapi.responses import Response, JSONResponse
from typing import Any, Dict
import base64

from app.models import RenderEnvelope
from app.pdf.shipment_confirmation import build_shipment_confirmation_pdf

app = FastAPI(title="ULP_PDF_PIPELINE", version="1.0")

@app.get("/health")
def health():
    return {"ok": True}

def _extract_request(payload: Any) -> Dict[str, Any]:
    """
    Accept either:
      1) Envelope: {"endpoint":..., "email_to":..., "request": {...}}
      2) Direct:   {...shipment request...}
    """
    if isinstance(payload, dict) and "request" in payload and isinstance(payload["request"], dict):
        return payload["request"]
    if isinstance(payload, dict):
        return payload
    return {}

@app.post("/api/v1/render/shipment-confirmation")
def render_shipment_confirmation(raw: Dict[str, Any] = Body(...)):
    req = _extract_request(raw)
    pdf_bytes = build_shipment_confirmation_pdf(req)

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": 'inline; filename="shipment_confirmation.pdf"'},
    )

@app.post("/api/v1/render/shipment-confirmation/base64")
def render_shipment_confirmation_base64(raw: Dict[str, Any] = Body(...)):
    req = _extract_request(raw)
    pdf_bytes = build_shipment_confirmation_pdf(req)

    return JSONResponse({
        "filename": "shipment_confirmation.pdf",
        "content_type": "application/pdf",
        "pdf_base64": base64.b64encode(pdf_bytes).decode("utf-8"),
    })
