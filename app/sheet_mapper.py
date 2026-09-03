from typing import Any, Dict, List, Optional


# ==========================================================
# EXACT GOOGLE SHEET HEADERS
# ==========================================================

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


# ==========================================================
# BASIC HELPERS
# ==========================================================

def _clean_value(
    value: Any,
):
    """
    Convert None to blank while preserving useful numeric/string
    values for the Sheet.
    """

    if value is None:
        return ""

    return value


def _highlight_for_confidence(
    confidence: Optional[float],
) -> Optional[str]:
    """
    Existing confidence rules:

        >= 80%       no highlight
        60% - <80%   yellow
        <60%         red
    """

    if confidence is None:
        return None

    try:
        confidence = float(
            confidence
        )

    except (
        TypeError,
        ValueError,
    ):
        return "red"

    if confidence < RED_THRESHOLD:
        return "red"

    if confidence < YELLOW_THRESHOLD:
        return "yellow"

    return None


# ==========================================================
# BACKWARD-COMPATIBLE FIELD HELPERS
# ==========================================================

def _field_value(
    field: Any,
):
    """
    Support BOTH structures.

    Old Document AI normalized field:

        {
            "value": "GILBERT",
            "confidence": 0.97,
            "page": 9
        }

    New hybrid field:

        "GILBERT"

    This lets the mapper remain backward-compatible.
    """

    if field is None:
        return None

    if isinstance(
        field,
        dict,
    ):

        return field.get(
            "value"
        )

    return field


def _field_confidence(
    field: Any,
) -> Optional[float]:
    """
    Old Document AI normalized fields may contain confidence.

    Hybrid GPT/Google fields currently do not have numeric
    confidence values, so they return None here.
    """

    if not isinstance(
        field,
        dict,
    ):
        return None

    confidence = field.get(
        "confidence"
    )

    if confidence is None:
        return None

    try:

        return float(
            confidence
        )

    except (
        TypeError,
        ValueError,
    ):

        return None


# ==========================================================
# COUNTRY
# ==========================================================

def _country_from_zip(
    zip_value: Any,
) -> Optional[str]:
    """
    Business rule:

    ZIP/postal starts with number -> USA
    ZIP/postal starts with letter -> CANADA
    """

    if zip_value in (
        None,
        "",
    ):
        return None

    text = str(
        zip_value
    ).strip()

    if not text:
        return None

    first = text[0]

    if first.isdigit():
        return "USA"

    if first.isalpha():
        return "CANADA"

    return None


# ==========================================================
# BLANK ROW / HIGHLIGHTS
# ==========================================================

def _make_blank_row() -> Dict[str, Any]:

    return {
        header: ""
        for header in SHEET_HEADERS
    }


def _make_blank_highlights(
) -> Dict[str, Optional[str]]:

    return {
        header: None
        for header in SHEET_HEADERS
    }


# ==========================================================
# CELL WRITER
# ==========================================================

def _set_field(
    row: Dict[str, Any],
    highlights: Dict[str, Optional[str]],
    column: str,
    value: Any,
    confidence: Optional[float] = None,
    required: bool = False,
    force_highlight: Optional[str] = None,
):
    """
    Write one Sheet field.

    Priority:

    1. Required + missing -> red
    2. Explicit force_highlight
    3. Numeric confidence thresholds
    """

    value = _clean_value(
        value
    )

    row[
        column
    ] = value

    if (
        required
        and value == ""
    ):

        highlights[
            column
        ] = "red"

        return

    if force_highlight:

        highlights[
            column
        ] = force_highlight

        return

    highlight = (
        _highlight_for_confidence(
            confidence
        )
    )

    if highlight:

        highlights[
            column
        ] = highlight


# ==========================================================
# SHIPMENT-LEVEL FIELDS
# ==========================================================

def _add_shipment_fields(
    row: Dict[str, Any],
    highlights: Dict[str, Optional[str]],
    order: Dict[str, Any],
):
    """
    Add shipment-level data.

    Called ONLY on the first row of a Sales Order.

    Hybrid output is normally:

        sales_order: "SO-00325428"
        customer_PO: "P-26.172JR-01"
        SRP_number: "SO0316672"
        delivery_name: "..."
        ...

    Old normalized Document AI dictionaries remain supported.
    """

    sales_order_source = (
        order.get(
            "sales_order"
        )
    )

    sales_order = _field_value(
        sales_order_source
    )

    sales_order_confidence = (
        order.get(
            "sales_order_confidence"
        )
    )

    if (
        sales_order_confidence
        is None
    ):

        sales_order_confidence = (
            _field_confidence(
                sales_order_source
            )
        )

    _set_field(
        row,
        highlights,
        "Sales Order",
        sales_order,
        confidence=sales_order_confidence,
        required=True,
    )

    field_map = {

        "delivery_name":
            "Delivery name",

        "delivery_address":
            "DEST_ADDRESS1_CLEAN",

        "delivery_address2":
            "DEST_ADDRESS2_CLEAN",

        "delivery_city":
            "City",

        "delivery_state":
            "State",

        "delivery_zip":
            "ZIP/postal code",

        "delivery_contact":
            "Delivery contact",

        "customer_PO":
            "Customer reference",

        "SRP_number":
            "SRP Number",
    }

    for (
        source_field,
        sheet_column,
    ) in field_map.items():

        source = order.get(
            source_field
        )

        value = _field_value(
            source
        )

        confidence = (
            _field_confidence(
                source
            )
        )

        _set_field(
            row,
            highlights,
            sheet_column,
            value,
            confidence=confidence,
            required=False,
        )

    # ------------------------------------------------------
    # COUNTRY
    # ------------------------------------------------------

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


# ==========================================================
# HANDLING UNIT
# ==========================================================

def _add_handling_unit(
    row: Dict[str, Any],
    highlights: Dict[str, Optional[str]],
    handling_unit: Dict[str, Any],
):
    """
    Write one handling unit.

    Required fields:

        L
        W
        H
        #
        Location

    Hybrid GPT behavior:

        uncertain = false
            -> normal cells

        uncertain = true
            -> populated HU cells yellow

        missing required field
            -> red

    Missing always wins over uncertainty.
    """

    confidence = handling_unit.get(
        "confidence",
        {},
    )

    if not isinstance(
        confidence,
        dict,
    ):
        confidence = {}

    uncertain = bool(
        handling_unit.get(
            "uncertain",
            False,
        )
    )

    mapping = {

        "length":
            "L",

        "width":
            "W",

        "height":
            "H",

        "weight":
            "#",

        "location":
            "Location",
    }

    for (
        source_field,
        sheet_column,
    ) in mapping.items():

        value = handling_unit.get(
            source_field
        )

        field_confidence = (
            confidence.get(
                source_field
            )
        )

        # GPT doesn't currently provide numeric field
        # confidence. If it explicitly marked the HU
        # uncertain, flag populated cells yellow.
        force_highlight = None

        if (
            uncertain
            and value not in (
                None,
                "",
            )
        ):

            force_highlight = (
                "yellow"
            )

        _set_field(
            row,
            highlights,
            sheet_column,
            value,
            confidence=field_confidence,
            required=True,
            force_highlight=force_highlight,
        )


# ==========================================================
# MAIN SHEET MAPPER
# ==========================================================

def build_sheet_rows(
    normalized_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convert normalized/hybrid ULP Sales Orders into Sheet rows.

    RULES

    FIRST ROW OF SHIPMENT
    ---------------------
    Contains:

        Sales Order
        address/contact information
        Customer Reference
        SRP Number
        first handling unit

    ADDITIONAL HU ROWS
    ------------------
    Contain only:

        L
        W
        H
        #
        Location

    NO HANDLING UNITS
    -----------------
    Still create one shipment row.

    L/W/H/#/Location will all be RED so the human operator
    can review the shipment.

    HIGHLIGHTING
    ------------

    Existing numeric confidence:

        >=80%       normal
        60-<80%     yellow
        <60%        red

    Hybrid GPT uncertainty:

        uncertain=false
            normal

        uncertain=true
            populated HU cells yellow

        missing required HU value
            red
    """

    sales_orders = (
        normalized_result.get(
            "sales_orders",
            []
        )
        or []
    )

    output_rows: List[
        Dict[str, Any]
    ] = []

    for order in sales_orders:

        if not isinstance(
            order,
            dict,
        ):
            continue

        handling_units = (
            order.get(
                "handling_units",
                []
            )
            or []
        )

        # ==================================================
        # NO HANDLING UNITS
        # ==================================================

        if not handling_units:

            row = (
                _make_blank_row()
            )

            highlights = (
                _make_blank_highlights()
            )

            _add_shipment_fields(
                row,
                highlights,
                order,
            )

            # Human review required.
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
                "values":
                    row,

                "highlights":
                    highlights,
            })

            continue

        # ==================================================
        # ONE ROW PER HANDLING UNIT
        # ==================================================

        for (
            index,
            handling_unit,
        ) in enumerate(
            handling_units
        ):

            if not isinstance(
                handling_unit,
                dict,
            ):
                continue

            row = (
                _make_blank_row()
            )

            highlights = (
                _make_blank_highlights()
            )

            # Only first HU row gets shipment/header data.
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
                "values":
                    row,

                "highlights":
                    highlights,
            })

    return {
        "headers":
            SHEET_HEADERS,

        "row_count":
            len(
                output_rows
            ),

        "rows":
            output_rows,
    }
