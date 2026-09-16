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

from app.packet_splitter import (
    analyze_packet,
    split_packet_to_zip,
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

        hybrid_result = (
            extract_ulp_with_gpt(
                pdf_bytes
            )
        )

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

        normalized = {
            "sales_orders":
                sales_orders
        }

        sheet_rows = (
            build_sheet_rows(
                normalized
            )
        )

        return JSONResponse({

            "ok":
                True,

            "filename":
                file.filename,

            "result":
                normalized,

            "sheet_rows":
                sheet_rows,

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


# ==========================================================
# ULP PACKET INSPECT
# ==========================================================

@app.post(
    "/api/v1/ulp/split-packet/inspect"
)
async def inspect_ulp_packet(
    file: UploadFile = File(...)
):

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
            analyze_packet(
                pdf_bytes
            )
        )

        return JSONResponse({

            "filename":
                file.filename,

            **result,
        })

    except Exception as exc:

        raise HTTPException(
            status_code=500,

            detail=(
                "Packet inspection failed: "
                f"{str(exc)}"
            ),
        )


# ==========================================================
# ULP PACKET SPLIT - PRODUCTION ZIP
# ==========================================================

@app.post(
    "/api/v1/ulp/split-packet"
)
async def split_ulp_packet(
    file: UploadFile = File(...)
):
    """
    SUCCESS:
        Return valid SO PDFs + manifest.json

    PARTIAL_SUCCESS:
        Return valid SO PDFs
        + REVIEW_REQUIRED_PAGES.pdf
        + manifest.json

    FAILED:
        Return JSON error information
        and no ZIP.
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
            split_packet_to_zip(
                pdf_bytes
            )
        )

        manifest = (
            result[
                "manifest"
            ]
        )

        status = (
            result[
                "status"
            ]
        )

        # ==================================================
        # TOTAL FAILURE
        # ==================================================

        if status == "FAILED":

            return JSONResponse(
                status_code=422,

                content={

                    "ok":
                        False,

                    "status":
                        "FAILED",

                    "filename":
                        file.filename,

                    "page_count":
                        manifest.get(
                            "page_count"
                        ),

                    "packet_count":
                        manifest.get(
                            "packet_count"
                        ),

                    "review_page_count":
                        manifest.get(
                            "review_page_count"
                        ),

                    "review_pages":
                        manifest.get(
                            "review_pages"
                        ),

                    "exception_count":
                        manifest.get(
                            "exception_count"
                        ),

                    "exceptions":
                        manifest.get(
                            "exceptions"
                        ),
                },
            )

        # ==================================================
        # SUCCESS OR PARTIAL SUCCESS
        # BOTH RETURN ZIP
        # ==================================================

        return Response(
            content=
                result[
                    "zip_bytes"
                ],

            media_type=
                "application/zip",

            headers={

                "Content-Disposition":
                    (
                        'attachment; '
                        'filename="ULP_Split_Packet.zip"'
                    ),

                "X-ULP-Status":
                    status,

                "X-ULP-Page-Count":
                    str(
                        manifest[
                            "page_count"
                        ]
                    ),

                "X-ULP-Packet-Count":
                    str(
                        manifest[
                            "packet_count"
                        ]
                    ),

                "X-ULP-Assigned-Page-Count":
                    str(
                        manifest[
                            "assigned_page_count"
                        ]
                    ),

                "X-ULP-Review-Page-Count":
                    str(
                        manifest[
                            "review_page_count"
                        ]
                    ),

                "X-ULP-Exception-Count":
                    str(
                        manifest[
                            "exception_count"
                        ]
                    ),
            },
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,

            detail=(
                "Packet split failed: "
                f"{str(exc)}"
            ),
        )
