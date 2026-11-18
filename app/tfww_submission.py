"""Validate reviewed freight and send one TFWW Shipment Import request."""

import json
import os
from datetime import datetime
from http.client import HTTPSConnection, HTTPException as HTTPClientError
from uuid import uuid4
from zoneinfo import ZoneInfo

from fastapi import HTTPException
from pydantic import BaseModel

from app.shipment_intake import Address, ShipmentFields, validate_shipment

ACCOUNTS = {
    "Covenant": (
        "TFWW_COVENANT_API_KEY", "Playcore- Covenant Logistics Group",
        "TForce Worldwide", "1000 Windham Parkway", "Bolingbrook", "60490",
    ),
    "Ultra Play": (
        "TFWW_ULTRA_PLAY_API_KEY", "Ultra Play Systems",
        "Ultra Play Systems", "1675 Locust St", "Red Bud", "62278",
    ),
}
SERVICE_CODES = {
    "lift gate destination": "LG1",
    "limited access delivery": "LADDOCK",
    "residential": "RES1",
    "appointment": "APPT",
}


class ReviewedShipment(BaseModel):
    account: str
    shipment: ShipmentFields


def map_address(address: Address, *, residential: bool = False) -> dict:
    return {
        "IsResidential": residential,
        "Name": address.name,
        "AddressLine1": address.address1,
        "AddressLine2": address.address2 or "",
        "City": address.city,
        "StateProvince": address.state,
        "PostalCode": address.postal_code,
        "CountryCode": address.country_code,
        "Contact": {
            "Name": address.contact_name or "",
            "Phone": address.phone or "",
            "Email": "",
        },
    }


def build_payload(review: ReviewedShipment, day: str) -> dict:
    shipment = review.shipment
    errors = validate_shipment(shipment, require_freight_class=True, require_nmfc=True)
    for role in ("shipper", "consignee"):
        address = getattr(shipment, role)
        for field in ("name", "address1", "city", "state", "country_code"):
            if not (getattr(address, field) or "").strip():
                errors.append(f"{role.capitalize()} {field} is required.")

    reference = shipment.reference
    candidates = [(reference.load_id, "Load ID"), (reference.sales_order, "Sales Order")]
    if review.account == "Ultra Play":
        candidates.reverse()
    candidates.append((reference.purchase_order, "Purchase Order"))
    references = [
        {"ReferenceNumber": value.strip(), "Type": kind, "IsPrimary": False}
        for value, kind in candidates if value and value.strip()
    ]
    if references:
        references[0]["IsPrimary"] = True
    else:
        errors.append("A shipment reference is required.")

    items = []
    for index, item in enumerate(shipment.items, 1):
        for field in ("description", "quantity_uom", "weight_uom"):
            if not (getattr(item, field) or "").strip():
                errors.append(f"Commodity {index} {field} is required.")
        if item.hazardous_material is None:
            errors.append(f"Confirm hazardous_material for commodity {index}.")
        try:
            freight_class = float(item.freight_class or "")
            if not 50 <= freight_class <= 500:
                raise ValueError
        except ValueError:
            freight_class = None
            errors.append(f"Commodity {index} freight class must be a number from 50 to 500.")
        mapped = {
            "Id": str(uuid4()),
            "Description": item.description,
            "FreightClasses": {"FreightClass": freight_class, "Type": ""},
            "NmfcCode": item.nmfc,
            "HazardousMaterial": item.hazardous_material,
            "Weights": {"Actual": item.weight, "Uom": item.weight_uom},
            "Quantities": {"Actual": item.quantity, "Uom": item.quantity_uom},
        }
        dimensions = (item.length, item.width, item.height)
        if any(value is not None for value in dimensions):
            if any(value is None or value <= 0 for value in dimensions) or not (item.dimension_uom or "").strip():
                errors.append(f"Commodity {index} dimensions require length, width, height and units.")
            else:
                mapped["Dimensions"] = {
                    "Length": item.length, "Width": item.width, "Height": item.height,
                    "Uom": item.dimension_uom,
                }
        items.append(mapped)
    if errors:
        raise HTTPException(422, errors)

    _, _, name, street, city, zipcode = ACCOUNTS[review.account]
    services = dict.fromkeys(
        SERVICE_CODES[label.strip().lower()] for label in shipment.source_services
        if label.strip().lower() in SERVICE_CODES
    )
    payload = {
        "Status": "Pending",
        "ReferenceNumbers": references,
        "Dates": {
            "EarliestPickupDate": f"{day} 08:00",
            "LatestPickupDate": f"{day} 15:00",
            "EarliestDropDate": f"{day} 08:00",
            "LatestDropDate": f"{day} 17:00",
        },
        "Shipper": map_address(shipment.shipper),
        "Consignee": map_address(shipment.consignee, residential="RES1" in services),
        "Items": items,
        "Payment": {"Address": {
            "Name": name, "AddressLine1": street, "City": city,
            "StateProvince": "IL", "PostalCode": zipcode, "CountryCode": "USA",
            "IsResidential": False, "Comments": "",
            "Contact": {"Name": "", "Email": "", "Phone": ""},
        }},
    }
    if services:
        payload["ServiceFlags"] = [
            {"ServiceCode": code, "IsSelected": True} for code in services
        ]
    return payload


def submit_to_tfww(review: ReviewedShipment) -> dict:
    if review.account not in ACCOUNTS:
        raise HTTPException(422, "Choose Covenant or Ultra Play.")
    day = datetime.now(ZoneInfo("America/Chicago")).strftime("%m/%d/%Y")
    payload = build_payload(review, day)
    env_name, username, *_ = ACCOUNTS[review.account]
    key = os.getenv(env_name)
    if not key:
        raise HTTPException(503, f"TFWW credentials are not configured for {review.account}.")

    connection = HTTPSConnection("connectapi.tfwwi.com", timeout=35)
    try:
        connection.request("POST", "/services/api/v1/Shipment/Import", json.dumps(payload), {
            "Content-Type": "application/json", "Accept": "application/json",
            "ApiKey": key, "UserName": username,
        })
        response = connection.getresponse()
        body = response.read(16000).decode("utf-8", errors="replace").replace(key, "[redacted]")
        return {"tfww_status": response.status, "response": body}
    except (OSError, HTTPClientError) as exc:
        raise HTTPException(502, "TFWW result could not be confirmed. Check TFWW before retrying.") from exc
    finally:
        connection.close()
