from collections import defaultdict
from typing import Any, Dict, List, Optional


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


PREFERRED_LENGTHS = {
    48,
    72,
    74,
    79,
    96,
    98,
    120,
    144,
}


def _best_entity(
    entities: List[Dict[str, Any]],
    field_type: str,
) -> Optional[Dict[str, Any]]:
    matches = [
        e
        for e in entities
        if e.get("type") == field_type
        and e.get("value") not in (None, "")
    ]

    if not matches:
        return None

    return max(
        matches,
        key=lambda e: float(e.get("confidence") or 0),
    )


def _field_result(
    entity: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not entity:
        return None

    return {
        "value": entity.get("value"),
        "confidence": float(entity.get("confidence") or 0),
        "page": entity.get("page"),
    }


def _numeric_value(value: Any):
    if value is None:
        return None

    text = str(value).strip().replace(",", "")

    try:
        number = float(text)

        if number.is_integer():
            return int(number)

        return number

    except (TypeError, ValueError):
        return None


def _numeric_entities_by_type(
    entities: List[Dict[str, Any]],
    field_type: str,
) -> List[Dict[str, Any]]:
    """
    Return only numeric dimension/weight entities.

    This deliberately rejects false OCR classifications like:
        length = "L"
    """

    results = []

    for entity in entities:
        if entity.get("type") != field_type:
            continue

        numeric = _numeric_value(
            entity.get("value")
        )

        if numeric is None:
            continue

        copied = dict(entity)
        copied["numeric_value"] = numeric

        results.append(copied)

    return results


def _entities_by_type(
    entities: List[Dict[str, Any]],
    field_type: str,
) -> List[Dict[str, Any]]:
    return [
        e
        for e in entities
        if e.get("type") == field_type
        and e.get("value") not in (None, "")
    ]


def _normalize_dimensions(
    length,
    width,
    height,
):
    """
    Apply ULP-specific dimension rules.

    Returns:
        normalized_length
        normalized_width
        normalized_height
        dimensions_adjusted
    """

    if not all(
        isinstance(v, (int, float))
        for v in [length, width, height]
    ):
        return length, width, height, False

    values = [
        length,
        width,
        height,
    ]

    #
    # Special case:
    # Two 48s plus another dimension.
    #
    # Example:
    #   20, 48, 48
    #
    # Expected:
    #   48 x 48 x 20
    #
    if values.count(48) == 2:

        remaining = list(values)

        remaining.remove(48)
        remaining.remove(48)

        other = remaining[0]

        normalized = (
            48,
            48,
            other,
        )

        adjusted = normalized != (
            length,
            width,
            height,
        )

        return (
            normalized[0],
            normalized[1],
            normalized[2],
            adjusted,
        )

    #
    # Preferred-length rule.
    #
    preferred_matches = [
        v
        for v in values
        if v in PREFERRED_LENGTHS
    ]

    if len(preferred_matches) == 1:

        preferred_length = preferred_matches[0]

        # Already correct.
        if length == preferred_length:
            return (
                length,
                width,
                height,
                False,
            )

        # Length and height were likely swapped.
        if height == preferred_length:
            return (
                height,
                width,
                length,
                True,
            )

        # Less common:
        # preferred length was classified as width.
        if width == preferred_length:
            return (
                width,
                length,
                height,
                True,
            )

    #
    # Ambiguous case:
    # Keep Document AI's original interpretation.
    #
    return (
        length,
        width,
        height,
        False,
    )


def _reassign_dimension_confidences(
    original_length,
    original_width,
    original_height,
    normalized_length,
    normalized_width,
    normalized_height,
    confidence,
):
    """
    Move confidence values with dimensions when dimensions
    are rearranged.
    """

    old = {
        "length": confidence.get("length", 0.0),
        "width": confidence.get("width", 0.0),
        "height": confidence.get("height", 0.0),
    }

    originals = [
        ("length", original_length),
        ("width", original_width),
        ("height", original_height),
    ]

    used = set()

    def confidence_for_value(value):
        for field_name, original_value in originals:

            if field_name in used:
                continue

            if original_value == value:
                used.add(field_name)
                return old[field_name]

        return 0.0

    confidence["length"] = confidence_for_value(
        normalized_length
    )

    confidence["width"] = confidence_for_value(
        normalized_width
    )

    confidence["height"] = confidence_for_value(
        normalized_height
    )


def _build_handling_units(
    entities: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Build individual skids / handling units.

    A page is treated as a measurement page only if Document AI
    extracted legitimate numeric dimension or weight data from it.
    """

    measurement_pages = set()

    for entity in entities:

        if entity.get("type") not in {
            "length",
            "width",
            "height",
            "weight",
        }:
            continue

        numeric = _numeric_value(
            entity.get("value")
        )

        if numeric is None:
            continue

        page = entity.get("page")

        if page is not None:
            measurement_pages.add(page)

    #
    # Ignore locations and other fields from ordinary product pages.
    #
    handling_entities = [
        e
        for e in entities
        if e.get("page") in measurement_pages
    ]

    lengths = _numeric_entities_by_type(
        handling_entities,
        "length",
    )

    widths = _numeric_entities_by_type(
        handling_entities,
        "width",
    )

    heights = _numeric_entities_by_type(
        handling_entities,
        "height",
    )

    weights = _numeric_entities_by_type(
        handling_entities,
        "weight",
    )

    locations = _entities_by_type(
        handling_entities,
        "location",
    )

    count = max(
        [
            len(lengths),
            len(widths),
            len(heights),
            len(weights),
            len(locations),
        ],
        default=0,
    )

    units = []

    for i in range(count):

        unit = {
            "handling_unit": i + 1,
        }

        confidence = {}
        pages = set()

        numeric_field_lists = {
            "length": lengths,
            "width": widths,
            "height": heights,
            "weight": weights,
        }

        #
        # Numeric fields.
        #
        for field_name, values in numeric_field_lists.items():

            if i < len(values):

                entity = values[i]

                unit[field_name] = entity[
                    "numeric_value"
                ]

                confidence[field_name] = float(
                    entity.get("confidence") or 0
                )

                if entity.get("page") is not None:
                    pages.add(
                        entity.get("page")
                    )

            else:

                unit[field_name] = None
                confidence[field_name] = 0.0

        #
        # Location.
        #
        if i < len(locations):

            location_entity = locations[i]

            unit["location"] = location_entity.get(
                "value"
            )

            confidence["location"] = float(
                location_entity.get(
                    "confidence"
                ) or 0
            )

            if location_entity.get("page") is not None:
                pages.add(
                    location_entity.get("page")
                )

        else:

            unit["location"] = None
            confidence["location"] = 0.0

        #
        # Normalize L/W/H.
        #
        original_length = unit.get("length")
        original_width = unit.get("width")
        original_height = unit.get("height")

        (
            normalized_length,
            normalized_width,
            normalized_height,
            dimensions_adjusted,
        ) = _normalize_dimensions(
            original_length,
            original_width,
            original_height,
        )

        if dimensions_adjusted:

            _reassign_dimension_confidences(
                original_length,
                original_width,
                original_height,
                normalized_length,
                normalized_width,
                normalized_height,
                confidence,
            )

        unit["length"] = normalized_length
        unit["width"] = normalized_width
        unit["height"] = normalized_height

        unit["dimensions_adjusted"] = (
            dimensions_adjusted
        )

        unit["confidence"] = confidence

        unit["pages"] = sorted(
            pages
        )

        present_confidences = [
            score
            for field, score in confidence.items()
            if unit.get(field) is not None
        ]

        if present_confidences:

            unit["minimum_confidence"] = min(
                present_confidences
            )

        else:

            unit["minimum_confidence"] = 0.0

        units.append(
            unit
        )

    return units


def normalize_ulp_document(
    extraction_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convert Document AI's flat entity list into structured
    Sales Orders and handling units.
    """

    entities = extraction_result.get(
        "entities",
        [],
    )

    sales_order_pages = defaultdict(set)

    #
    # Determine which pages belong to each Sales Order.
    #
    for entity in entities:

        if entity.get("type") != "sales_order":
            continue

        sales_order = str(
            entity.get("value") or ""
        ).strip()

        page = entity.get(
            "page"
        )

        if (
            sales_order
            and page is not None
        ):

            sales_order_pages[
                sales_order
            ].add(
                page
            )

    sales_orders = []

    #
    # Build each Sales Order.
    #
    for sales_order, pages in sales_order_pages.items():

        order_entities = [
            e
            for e in entities
            if e.get("page") in pages
        ]

        order = {
            "sales_order": sales_order,
            "pages": sorted(pages),
        }

        #
        # Sales Order confidence.
        #
        so_entities = [
            e
            for e in order_entities
            if (
                e.get("type") == "sales_order"
                and str(
                    e.get("value") or ""
                ).strip() == sales_order
            )
        ]

        if so_entities:

            best_so = max(
                so_entities,
                key=lambda e: float(
                    e.get("confidence") or 0
                ),
            )

            order[
                "sales_order_confidence"
            ] = float(
                best_so.get(
                    "confidence"
                ) or 0
            )

        else:

            order[
                "sales_order_confidence"
            ] = 0.0

        #
        # Shipment-level information.
        #
        for field_name in SHIPMENT_FIELDS:

            best = _best_entity(
                order_entities,
                field_name,
            )

            order[field_name] = (
                _field_result(
                    best
                )
            )

        #
        # Individual skids.
        #
        order[
            "handling_units"
        ] = _build_handling_units(
            order_entities
        )

        sales_orders.append(
            order
        )

    #
    # Preserve document order.
    #
    sales_orders.sort(
        key=lambda order: (
            order["pages"][0]
            if order.get("pages")
            else 999999
        )
    )

    return {
        "sales_order_count": len(
            sales_orders
        ),
        "sales_orders": sales_orders,
    }
