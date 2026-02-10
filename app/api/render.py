from __future__ import annotations

from fastapi import APIRouter, Body, Response
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.models.shipment import Shipment
from app.renderers.shipment_confirmation import render_shipment_confirmation_pdf

router = APIRouter(tags=["render"])


@router.post("/render/shipment-confirmation")
def render_shipment_confirmation(payload: dict = Body(...)):
    """
    Accepts your shipment JSON, renders a Shipment Confirmation PDF.
    Returns application/pdf bytes.
    """
    try:
        shipment = Shipment.model_validate(payload)
    except ValidationError as e:
        # You want Sheets-friendly errors: return structured JSON
        return JSONResponse(
            status_code=422,
            content={
                "error": "VALIDATION_ERROR",
                "details": e.errors(),
            },
        )

    pdf_bytes = render_shipment_confirmation_pdf(shipment)

    # Stream PDF back
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'inline; filename="shipment_confirmation_{shipment.primary_reference() or "doc"}.pdf"'
        },
    )
