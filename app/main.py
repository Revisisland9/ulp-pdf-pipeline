from fastapi import FastAPI, Body, UploadFile, File, HTTPException
from fastapi.responses import Response, JSONResponse
from typing import Any, Dict
import base64

from app.models import RenderEnvelope
from app.pdf.shipment_confirmation import build_shipment_confirmation_pdf
from app.document_ai import extract_ulp_document


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
    if (
        isinstance(payload, dict)
        and "request" in payload
        and isinstance(payload["request"], dict)
    ):
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
        headers={
            "Content-Disposition":
            'inline; filename="shipment_confirmation.pdf"'
        },
    )


@app.post("/api/v1/render/shipment-confirmation/base64")
def render_shipment_confirmation_base64(
    raw: Dict[str, Any] = Body(...)
):
    req = _extract_request(raw)

    pdf_bytes = build_shipment_confirmation_pdf(req)

    return JSONResponse({
        "filename": "shipment_confirmation.pdf",
        "content_type": "application/pdf",
        "pdf_base64": base64.b64encode(pdf_bytes).decode("utf-8"),
    })


@app.post("/api/v1/ulp/extract")
async def extract_ulp_pdf(
    file: UploadFile = File(...)
):
    """
    Accept a ULP PDF and send it to the
    Google Document AI ULP Custom Extractor.

    For the current processor, PDFs must be
    15 pages or fewer.
    """

    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="File must be a PDF."
        )

    pdf_bytes = await file.read()

    if not pdf_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded PDF is empty."
        )

    try:
        result = extract_ulp_document(pdf_bytes)

        return JSONResponse({
            "ok": True,
            "filename": file.filename,
            "result": result
        })

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Document extraction failed: {str(exc)}"
        )
