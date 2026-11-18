"""Authenticated extraction endpoint for the shipment intake sidebar."""

import hmac
import logging
import os

from fastapi import APIRouter, Depends, File, Header, HTTPException, UploadFile

from app.shipment_intake import (
    MAX_DOCUMENT_BYTES,
    DocumentInputError,
    ShipmentExtraction,
    parse_document_to_shipment,
)

from app.tfww_submission import ReviewedShipment, submit_to_tfww

router = APIRouter()
logger = logging.getLogger(__name__)


def require_intake_token(x_shipment_intake_token: str = Header(default="")) -> None:
    expected = os.getenv("SHIPMENT_INTAKE_TOKEN")
    if not expected:
        raise HTTPException(503, "Shipment intake is not configured.")
    if not hmac.compare_digest(expected.encode(), x_shipment_intake_token.encode()):
        raise HTTPException(401, "Invalid shipment intake token.")


@router.post("/shipment-intake/extract", response_model=ShipmentExtraction,
             dependencies=[Depends(require_intake_token)])
def extract_shipment(file: UploadFile = File(...)) -> ShipmentExtraction:
    try:
        data = file.file.read(MAX_DOCUMENT_BYTES + 1)
        return parse_document_to_shipment(file.filename or "", data)
    except DocumentInputError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        logger.error("Shipment extraction failed (%s)", type(exc).__name__)
        raise HTTPException(502, "Shipment extraction failed. Please try again.") from exc


@router.post("/shipment-intake/submit", dependencies=[Depends(require_intake_token)])
def submit_shipment(review: ReviewedShipment) -> dict:
    """Submit reviewed freight as non-hazardous freight after validating dimension units."""
    return submit_to_tfww(review)
