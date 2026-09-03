from fastapi import FastAPI, Body, UploadFile, File, HTTPException
from fastapi.responses import Response, JSONResponse
from typing import Any, Dict
import base64

from app.models import RenderEnvelope
from app.pdf.shipment_confirmation import build_shipment_confirmation_pdf

from app.document_ai import extract_ulp_document
from app.ulp_normalizer import normalize_ulp_document
from app.sheet_mapper import build_sheet_rows

from app.openai_extractor import extract_ulp_with_gpt


app = FastAPI(
    title="ULP_PDF_PIPELINE",
    version="1.0",
)


# ==========================================================
# HEALTH
# ==========================================================

@app.get("/health")
def health():
    return {
        "ok": True
    }


# ==========================================================
# REQUEST HELPER
# ==========================================================

def _extract_request(
    payload: Any
) -> Dict[str, Any]:
    """
    Accept either:

    1) Envelope:
       {
         "endpoint": ...,
         "email_to": ...,
         "request": {...}
       }

    2) Direct:
       {...shipment request...}
    """

    if (
        isinstance(payload, dict)
        and "request" in payload
        and isinstance(
            payload["request"],
            dict,
        )
    ):
        return payload["request"]

    if isinstance(payload, dict):
        return payload

    return {}


# ==========================================================
# SHIPMENT CONFIRMATION PDF
# ==========================================================

@app.post(
    "/api/v1/render/shipment-confirmation"
)
def render_shipment_confirmation(
    raw: Dict[str, Any] = Body(...)
):
    req = _extract_request(
        raw
    )

    pdf_bytes = (
        build_shipment_confirmation_pdf(
            req
        )
    )

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition":
            'inline; filename="shipment_confirmation.pdf"'
        },
    )


# ==========================================================
# SHIPMENT CONFIRMATION PDF - BASE64
# ==========================================================

@app.post(
    "/api/v1/render/shipment-confirmation/base64"
)
def render_shipment_confirmation_base64(
    raw: Dict[str, Any] = Body(...)
):
    req = _extract_request(
        raw
    )

    pdf_bytes = (
        build_shipment_confirmation_pdf(
            req
        )
    )

    return JSONResponse({
        "filename":
            "shipment_confirmation.pdf",

        "content_type":
            "application/pdf",

        "pdf_base64":
            base64.b64encode(
                pdf_bytes
            ).decode("utf-8"),
    })


# ==========================================================
# EXISTING GOOGLE DOCUMENT AI ENDPOINT
# ==========================================================

@app.post(
    "/api/v1/ulp/extract"
)
async def extract_ulp_pdf(
    file: UploadFile = File(...)
):
    """
    CURRENT PRODUCTION ULP WORKFLOW:

    PDF
      -> Google Document AI
      -> normalized Sales Orders
      -> Sheet-ready rows
      -> highlight instructions

    This endpoint remains unchanged while
    the GPT extractor is tested separately.
    """

    if (
        file.content_type
        != "application/pdf"
    ):
        raise HTTPException(
            status_code=400,
            detail="File must be a PDF.",
        )

    pdf_bytes = await file.read()

    if not pdf_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded PDF is empty.",
        )

    try:

        #
        # Step 1
        # Google Document AI extraction
        #
        raw_result = (
            extract_ulp_document(
                pdf_bytes
            )
        )

        #
        # Step 2
        # Normalize extracted entities
        #
        normalized = (
            normalize_ulp_document(
                raw_result
            )
        )

        #
        # Step 3
        # Convert to Sheet-ready rows
        #
        sheet_rows = (
            build_sheet_rows(
                normalized
            )
        )

        return JSONResponse({
            "ok": True,
            "filename": file.filename,

            "result":
                normalized,

            "sheet_rows":
                sheet_rows,
        })

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Document extraction failed: "
                f"{str(exc)}"
            ),
        )


# ==========================================================
# GPT TEST ENDPOINT
# ==========================================================

@app.post(
    "/api/v1/ulp/extract-gpt"
)
async def extract_ulp_gpt_test(
    file: UploadFile = File(...)
):
    """
    TEMPORARY GPT TEST ENDPOINT.

    This does NOT replace the existing
    Google Document AI endpoint.

    For now, this endpoint verifies that:

    - Cloud Run can access OPENAI_API_KEY
    - the OpenAI Python SDK is installed
    - Cloud Run can successfully call OpenAI
    - PDF upload routing works

    The actual PDF / vision extraction logic
    will be added to openai_extractor.py
    after this connection test succeeds.
    """

    if (
        file.content_type
        != "application/pdf"
    ):
        raise HTTPException(
            status_code=400,
            detail="File must be a PDF.",
        )

    pdf_bytes = await file.read()

    if not pdf_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded PDF is empty.",
        )

    try:

        result = (
            extract_ulp_with_gpt(
                pdf_bytes
            )
        )

        return JSONResponse({
            "ok": True,
            "filename":
                file.filename,

            "result":
                result,
        })

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "GPT extraction test failed: "
                f"{str(exc)}"
            ),
        )
