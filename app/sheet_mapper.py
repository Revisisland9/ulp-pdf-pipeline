from typing import Any, Dict, List, Optional


# Exact Google Sheet headers, left to right.
SHEET_HEADERS = [
    "Sales Order",
    "THR",
    "THR PRO",
    "QA",
    "Knoxville",
    "SRP",
    "Belson",
    "Residential",
    "Limited Access",
    "Liftgate",
    "L",
    "W",
    "H",
    "#",
    "Location",
    "CL",
    "NMFC",
    "OVL",
    "Linear Ft.",
    "Delivery name",
    "DEST_ADDRESS1_CLEAN",
    "DEST_ADDRESS2_CLEAN",
    "City",
    "State",
    "ZIP/postal code",
    "Country/region",
    "Delivery contact",
    "Customer reference",
    "SRP Number",
    "THR Carrier",
    "Rated At",
    "Rate Status",
    "Selected SCAC",
    "Selected Carrier",
    "Selected Service",
    "Selected Service Days",
    "Selected Total",
    "Considered Rates",
    "Filtered Out Rates",
    "Rate Options (JSON)",
    "API Returned Ref",
]


YELLOW_THRESHOLD = 0.80
RED_THRESHOLD = 0.60


def _highlight_for_confidence(
    confidence: Optional[float],
) -> Optional[str]:
    """
    Return:
      None     = no highlight
      yellow   = confidence < 80% and >= 60%
      red      = confidence < 60%
    """

    if confidence is None:
        return None

    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        return "red"

    if confidence < RED_THRESHOLD:
        return "red"

    if confidence < YELLOW_THRESHOLD:
        return "yellow"

    return None


def _field_value(
    field: Optional[Dict[str, Any]],
):
    """
    Shipment-level normalized fields look like:

    {
        "value": "...",
        "confidence": 0.99,
        "page": 1
    }
    """

    if not field:
        return None

    return field.get("value")


def _field_confidence(
    field: Optional[Dict[str, Any]],
) -> Optional[float]:

    if not field:
        return None

    confidence = field.get("confidence")

    if confidence is None:
        return None

    try:
        return float(confidence)
    except (TypeError, ValueError):
        return None


def _country_from_zip(
    zip_value: Any,
) -> Optional[str]:
    """
    Business rule:

    Numeric-leading ZIP/postal code -> USA
    Letter-leading postal code      -> CANADA
    """

    if zip_value in (None, ""):
        return None

    text = str(zip_value).strip()

    if not text:
        return None

    first = text[0]

    if first.isdigit():
        return "USA"

    if first.isalpha():
        return "CANADA"

    return None


def _make_blank_row() -> Dict[str, Any]:
    """
    Create one completely blank row using the exact Sheet headers.
    """

    return {
        header: ""
        for header in SHEET_HEADERS
    }


def _make_blank_highlights() -> Dict[str, Optional[str]]:
    """
    Parallel formatting map.

    Example:
    {
        "L": "yellow",
        "Location": "red"
    }
    """

    return {
        header: None
        for header in SHEET_HEADERS
    }


def _set_field(
    row: Dict[str, Any],
    highlights: Dict[str, Optional[str]],
    column: str,
    value: Any,
    confidence: Optional[float] = None,
    required: bool = False,
):
    """
    Write a value and determine the cell highlight.

    Required + missing -> red.
    Otherwise confidence thresholds apply.
    """

    if value is None:
        value = ""

    row[column] = value

    if required and value == "":
        highlights[column] = "red"
        return

    highlight = _highlight_for_confidence(
        confidence
    )

    if highlight:
        highlights[column] = highlight


def _add_shipment_fields(
    row: Dict[str, Any],
    highlights: Dict[str, Optional[str]],
    order: Dict[str, Any],
):
    """
    Add Sales Order and address/contact data.

    This is only called for the FIRST row of each Sales Order.
    """

    sales_order = order.get(
        "sales_order",
        "",
    )

    sales_order_confidence = order.get(
        "sales_order_confidence"
    )

    _set_field(
        row,
        highlights,
        "Sales Order",
        sales_order,
        sales_order_confidence,
        required=True,
    )

    field_map = {
        "delivery_name": "Delivery name",
        "delivery_address": "DEST_ADDRESS1_CLEAN",
        "delivery_address2": "DEST_ADDRESS2_CLEAN",
        "delivery_city": "City",
        "delivery_state": "State",
        "delivery_zip": "ZIP/postal code",
        "delivery_contact": "Delivery contact",
        "customer_PO": "Customer reference",
        "SRP_number": "SRP Number",
    }

    for source_field, sheet_column in field_map.items():

        source = order.get(
            source_field
        )

        value = _field_value(
            source
        )

        confidence = _field_confidence(
            source
        )

        _set_field(
            row,
            highlights,
            sheet_column,
            value,
            confidence,
            required=False,
        )

    zip_value = row.get(
        "ZIP/postal code",
        ""
    )

    country = _country_from_zip(
        zip_value
    )

    _set_field(
        row,
        highlights,
        "Country/region",
        country,
        confidence=None,
        required=False,
    )


def _add_handling_unit(
    row: Dict[str, Any],
    highlights: Dict[str, Optional[str]],
    handling_unit: Dict[str, Any],
):
    """
    Add handling-unit values.

    L, W, H, weight and Location are all required.
    """

    confidence = handling_unit.get(
        "confidence",
        {},
    )

    mapping = {
        "length": "L",
        "width": "W",
        "height": "H",
        "weight": "#",
        "location": "Location",
    }

    for source_field, sheet_column in mapping.items():

        value = handling_unit.get(
            source_field
        )

        field_confidence = confidence.get(
            source_field
        )

        _set_field(
            row,
            highlights,
            sheet_column,
            value,
            field_confidence,
            required=True,
        )


def build_sheet_rows(
    normalized_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convert normalized ULP Sales Orders into Sheet-ready rows.

    Rules:

    - First handling-unit row gets Sales Order + address/contact data.
    - Additional handling-unit rows contain only L/W/H/#/Location.
    - All unrelated columns remain blank.
    - Missing required handling-unit fields are red.
    - Confidence:
        >= 80%          no highlight
        60% - <80%      yellow
        <60%            red
    - If a Sales Order has NO handling units:
        create one shipment row and mark
        L/W/H/#/Location red.
    """

    sales_orders = normalized_result.get(
        "sales_orders",
        [],
    )

    output_rows: List[Dict[str, Any]] = []

    for order in sales_orders:

        handling_units = order.get(
            "handling_units",
            [],
        )

        #
        # No handling units:
        # preserve the Sales Order row but make the required
        # handling-unit cells red.
        #
        if not handling_units:

            row = _make_blank_row()
            highlights = _make_blank_highlights()

            _add_shipment_fields(
                row,
                highlights,
                order,
            )

            for required_column in [
                "L",
                "W",
                "H",
                "#",
                "Location",
            ]:
                highlights[
                    required_column
                ] = "red"

            output_rows.append({
                "values": row,
                "highlights": highlights,
            })

            continue

        #
        # Normal case:
        # one row per handling unit.
        #
        for index, handling_unit in enumerate(
            handling_units
        ):

            row = _make_blank_row()
            highlights = _make_blank_highlights()

            #
            # Only the first handling-unit row gets
            # Sales Order/address/contact data.
            #
            if index == 0:

                _add_shipment_fields(
                    row,
                    highlights,
                    order,
                )

            _add_handling_unit(
                row,
                highlights,
                handling_unit,
            )

            output_rows.append({
                "values": row,
                "highlights": highlights,
            })

    return {
        "headers": SHEET_HEADERS,
        "row_count": len(output_rows),
        "rows": output_rows,
    }
