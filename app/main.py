from fastapi import FastAPI, Body, UploadFile, File, HTTPException
from fastapi.responses import Response, JSONResponse
from typing import Any, Dict
import base64

from app.models import RenderEnvelope
from app.pdf.shipment_confirmation import build_shipment_confirmation_pdf

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
# PRODUCTION ULP INTAKE
# ==========================================================

@app.post(
    "/api/v1/ulp/extract"
)
async def extract_ulp_pdf(
    file: UploadFile = File(...)
):
    """
    PRODUCTION ULP WORKFLOW:

    PDF
      -> Google Enterprise OCR
      -> GPT vision / handwriting extraction
      -> Python grouping + validation
      -> Sheet-ready rows
      -> existing Apps Script workflow

    Apps Script continues calling the SAME endpoint.
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

        # --------------------------------------------------
        # STEP 1
        # Hybrid Google OCR + GPT extraction
        # --------------------------------------------------

        hybrid_result = (
            extract_ulp_with_gpt(
                pdf_bytes
            )
        )

        # --------------------------------------------------
        # STEP 2
        # Pull grouped shipment records from hybrid result
        # --------------------------------------------------

        normalized = {
            "sales_orders":
                (
                    hybrid_result
                    .get(
                        "extraction",
                        {}
                    )
                    .get(
                        "sales_orders",
                        []
                    )
                )
        }

        # --------------------------------------------------
        # STEP 3
        # Convert to existing Sheet-ready format
        # --------------------------------------------------

        sheet_rows = (
            build_sheet_rows(
                normalized
            )
        )

        # --------------------------------------------------
        # STEP 4
        # Preserve existing production response contract
        # --------------------------------------------------

        return JSONResponse({
            "ok":
                True,

            "filename":
                file.filename,

            "result":
                normalized,

            "sheet_rows":
                sheet_rows,

            # Helpful production diagnostics.
            # Apps Script can ignore these.
            "engine":
                "google_ocr_plus_gpt",

            "usage":
                hybrid_result.get(
                    "usage"
                ),

            "google_ocr":
                hybrid_result.get(
                    "google_ocr"
                ),
        })

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Hybrid document extraction failed: "
                f"{str(exc)}"
            ),
        )


# ==========================================================
# GPT / HYBRID TEST ENDPOINT
# ==========================================================

@app.post(
    "/api/v1/ulp/extract-gpt"
)
async def extract_ulp_gpt_test(
    file: UploadFile = File(...)
):
    """
    Diagnostic endpoint.

    Returns the full hybrid result including:
    - page debug
    - Google OCR Sales Orders
    - GPT extraction
    - token usage
    - grouped shipments

    Keep this endpoint available for troubleshooting.
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
            "ok":
                True,

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
