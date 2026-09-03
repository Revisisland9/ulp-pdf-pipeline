from fastapi import (
    FastAPI,
    Body,
    UploadFile,
    File,
    HTTPException,
)

from fastapi.responses import (
    Response,
    JSONResponse,
)

from typing import Any, Dict

import base64


from app.models import RenderEnvelope

from app.pdf.shipment_confirmation import (
    build_shipment_confirmation_pdf
)

from app.sheet_mapper import (
    build_sheet_rows
)

from app.openai_extractor import (
    extract_ulp_with_gpt
)


app = FastAPI(
    title="ULP_PDF_PIPELINE",
    version="1.0",
)


# ==========================================================
# HEALTH
# ==========================================================

@app.get(
    "/health"
)
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

    1) Envelope

       {
           "endpoint": ...,
           "email_to": ...,
           "request": {...}
       }

    2) Direct shipment request

       {...}
    """

    if (
        isinstance(
            payload,
            dict,
        )
        and "request" in payload
        and isinstance(
            payload[
                "request"
            ],
            dict,
        )
    ):

        return payload[
            "request"
        ]

    if isinstance(
        payload,
        dict,
    ):

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

        media_type=
            "application/pdf",

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
            ).decode(
                "utf-8"
            ),
    })


# ==========================================================
# PRODUCTION ULP PINK INTAKE
# ==========================================================

@app.post(
    "/api/v1/ulp/extract"
)
async def extract_ulp_pdf(
    file: UploadFile = File(...)
):
    """
    PRODUCTION PINK WORKFLOW

    PDF
        ↓
    Google Enterprise OCR
        ↓
    GPT vision / handwriting interpretation
        ↓
    Python validation + shipment grouping
        ↓
    Sheet mapper
        ↓
    existing Apps Script intake

    IMPORTANT:

    Apps Script continues calling exactly the same endpoint:

        /api/v1/ulp/extract
    """

    # ------------------------------------------------------
    # VALIDATE FILE
    # ------------------------------------------------------

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

        # ==================================================
        # STEP 1
        # HYBRID EXTRACTION
        # ==================================================

        hybrid_result = (
            extract_ulp_with_gpt(
                pdf_bytes
            )
        )

        # ==================================================
        # STEP 2
        # GET GROUPED SALES ORDERS
        # ==================================================

        extraction = (
            hybrid_result.get(
                "extraction",
                {}
            )
            or {}
        )

        sales_orders = (
            extraction.get(
                "sales_orders",
                []
            )
            or []
        )

        # ==================================================
        # STEP 3
        # CREATE NORMALIZED PRODUCTION OBJECT
        # ==================================================

        normalized = {
            "sales_orders":
                sales_orders
        }

        # ==================================================
        # STEP 4
        # BUILD EXACT EXISTING SHEET CONTRACT
        # ==================================================

        sheet_rows = (
            build_sheet_rows(
                normalized
            )
        )

        # ==================================================
        # PRODUCTION RESPONSE
        # ==================================================

        return JSONResponse({

            "ok":
                True,

            "filename":
                file.filename,

            # Keep same key Apps Script already expects.
            "result":
                normalized,

            # Keep same Sheet response contract.
            "sheet_rows":
                sheet_rows,

            # Additional diagnostics.
            # Apps Script can simply ignore these.
            "engine":
                "google_ocr_plus_gpt",

            "google_ocr":
                hybrid_result.get(
                    "google_ocr"
                ),

            "usage":
                hybrid_result.get(
                    "usage"
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
# HYBRID DEBUG / TEST ENDPOINT
# ==========================================================

@app.post(
    "/api/v1/ulp/extract-gpt"
)
async def extract_ulp_gpt_test(
    file: UploadFile = File(...)
):
    """
    Diagnostic endpoint.

    Unlike production, this returns the complete extraction
    including per-page diagnostics.

    Keep this around for troubleshooting new Pink formats.
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
