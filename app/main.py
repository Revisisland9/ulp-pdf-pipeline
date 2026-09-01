from fastapi import FastAPI, Body, UploadFile, File, HTTPException
from fastapi.responses import Response, JSONResponse
from typing import Any, Dict
import base64

from app.models import RenderEnvelope
from app.pdf.shipment_confirmation import build_shipment_confirmation_pdf
from app.document_ai import extract_ulp_document
from app.ulp_normalizer import normalize_ulp_document


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
    Accept a ULP PDF, send it to Google Document AI,
    and normalize the extracted entities into
    Sales Orders and handling units.

    Current synchronous Document AI processor limit:
    15 pages per PDF.
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
        # Step 1:
        # Send the PDF to Google Document AI.
        raw_result = extract_ulp_document(pdf_bytes)

        # Step 2:
        # Turn Google's flat entity list into
        # structured Sales Orders / handling units.
        normalized = normalize_ulp_document(raw_result)

        return JSONResponse({
            "ok": True,
            "filename": file.filename,
            "result": normalized
        })

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Document extraction failed: {str(exc)}"
        )
