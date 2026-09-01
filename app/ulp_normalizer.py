from collections import defaultdict
from typing import Any, Dict, List, Optional


# Shipment-level fields.
# These normally describe the entire Sales Order rather than an individual skid.
SHIPMENT_FIELDS = {
    "customer_PO",
    "SRP_number",
    "delivery_name",
    "delivery_address",
    "delivery_address2",
    "delivery_city",
    "delivery_state",
    "delivery_zip",
    "delivery_contact",
}


# These fields describe individual handling units / skids.
HANDLING_UNIT_FIELDS = {
    "length",
    "width",
    "height",
    "weight",
    "location",
}


def _best_entity(
    entities: List[Dict[str, Any]],
    field_type: str,
) -> Optional[Dict[str, Any]]:
    """
    Return the highest-confidence occurrence of a field.
    """
    matches = [
        e for e in entities
        if e.get("type") == field_type
        and e.get("value") not in (None, "")
    ]

    if not matches:
        return None

    return max(
        matches,
        key=lambda e: float(e.get("confidence") or 0)
    )


def _field_result(
    entity: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """
    Convert an entity into the format we want to return.
    """
    if not entity:
        return None

    return {
        "value": entity.get("value"),
        "confidence": float(entity.get("confidence") or 0),
        "page": entity.get("page"),
    }


def _entities_by_type(
    entities: List[Dict[str, Any]],
    field_type: str,
) -> List[Dict[str, Any]]:
    """
    Return all occurrences of a field in document order.
    """
    return [
        e for e in entities
        if e.get("type") == field_type
        and e.get("value") not in (None, "")
    ]


def _numeric_value(value: Any):
    """
    Convert a numeric-looking field to int/float when possible.

    Bad extractions such as length='L' remain strings so the
    Google Sheet can flag them for human review.
    """
    if value is None:
        return None

    text = str(value).strip().replace(",", "")

    try:
        number = float(text)

        if number.is_integer():
            return int(number)

        return number

    except (TypeError, ValueError):
        return value


def _build_handling_units(
    entities: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Build handling units from repeated dimension / weight / location fields.

    Important:
    We only use locations from pages containing actual dimension or
    weight data. This prevents false-positive locations such as
    DSE, G-K, Inst., etc. from printed item tables.
    """

    # Find pages that clearly contain handling-unit measurements.
    measurement_pages = set()

    for entity in entities:
        if entity.get("type") in {
            "length",
            "width",
            "height",
            "weight",
        }:
            page = entity.get("page")
            if page is not None:
                measurement_pages.add(page)

    handling_entities = [
        e for e in entities
        if e.get("page") in measurement_pages
    ]

    lengths = _entities_by_type(handling_entities, "length")
    widths = _entities_by_type(handling_entities, "width")
    heights = _entities_by_type(handling_entities, "height")
    weights = _entities_by_type(handling_entities, "weight")
    locations = _entities_by_type(handling_entities, "location")

count = max([
    len(lengths),
    len(widths),
    len(heights),
    len(weights),
    len(locations),
], default=0)

    units = []

    for i in range(count):
        unit = {
            "handling_unit": i + 1
        }

        confidence = {}
        pages = set()

        field_lists = {
            "length": lengths,
            "width": widths,
            "height": heights,
            "weight": weights,
            "location": locations,
        }

        for field_name, values in field_lists.items():
            if i < len(values):
                entity = values[i]

                raw_value = entity.get("value")

                if field_name in {
                    "length",
                    "width",
                    "height",
                    "weight",
                }:
                    value = _numeric_value(raw_value)
                else:
                    value = raw_value

                unit[field_name] = value

                confidence[field_name] = float(
                    entity.get("confidence") or 0
                )

                if entity.get("page") is not None:
                    pages.add(entity["page"])

            else:
                unit[field_name] = None
                confidence[field_name] = 0.0

        unit["confidence"] = confidence
        unit["pages"] = sorted(pages)

        # Lowest-confidence extracted field gives us an easy
        # handling-unit review score.
        present_confidences = [
            score
            for field, score in confidence.items()
            if unit.get(field) is not None
        ]

        unit["minimum_confidence"] = (
            min(present_confidences)
            if present_confidences
            else 0.0
        )

        units.append(unit)

    return units


def normalize_ulp_document(
    extraction_result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Convert Document AI's flat entity list into:

        Sales Order
          ├─ shipment-level information
          └─ handling units
               ├─ L/W/H
               ├─ weight
               └─ location
    """

    entities = extraction_result.get("entities", [])

    # Determine which Sales Orders occur on which pages.
    sales_order_pages = defaultdict(set)

    for entity in entities:
        if entity.get("type") != "sales_order":
            continue

        sales_order = str(
            entity.get("value") or ""
        ).strip()

        page = entity.get("page")

        if sales_order and page is not None:
            sales_order_pages[sales_order].add(page)

    sales_orders = []

    for sales_order, pages in sales_order_pages.items():

        # Keep only entities occurring on pages belonging to this SO.
        order_entities = [
            e for e in entities
            if e.get("page") in pages
        ]

        order = {
            "sales_order": sales_order,
            "pages": sorted(pages),
        }

        # Sales Order confidence itself.
        so_entities = [
            e for e in order_entities
            if e.get("type") == "sales_order"
            and str(e.get("value") or "").strip() == sales_order
        ]

        if so_entities:
            best_so = max(
                so_entities,
                key=lambda e: float(
                    e.get("confidence") or 0
                )
            )

            order["sales_order_confidence"] = float(
                best_so.get("confidence") or 0
            )

        # Shipment-level fields.
        for field_name in SHIPMENT_FIELDS:
            best = _best_entity(
                order_entities,
                field_name,
            )

            order[field_name] = _field_result(best)

        # Individual skids / handling units.
        order["handling_units"] = _build_handling_units(
            order_entities
        )

        sales_orders.append(order)

    # Sort by first page appearance.
    sales_orders.sort(
        key=lambda order: (
            order["pages"][0]
            if order.get("pages")
            else 999999
        )
    )

    return {
        "sales_order_count": len(sales_orders),
        "sales_orders": sales_orders,
    }
