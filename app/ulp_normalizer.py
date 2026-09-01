from collections import defaultdict
from typing import Any, Dict, List, Optional


# Shipment-level fields.
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


# Handling-unit fields.
HANDLING_UNIT_FIELDS = {
    "length",
    "width",
    "height",
    "weight",
    "location",
}


# Based on ULP's normal freight / skid dimensions,
# these values are very commonly the LENGTH.
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
    """
    Return the highest-confidence occurrence of a field.
    """

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
    """
    Convert an entity into the normalized output format.
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
        e
        for e in entities
        if e.get("type") == field_type
        and e.get("value") not in (None, "")
    ]


def _numeric_value(value: Any):
    """
    Convert numeric-looking values to int/float.
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


def _normalize_dimensions(
    length,
    width,
    height,
):
    """
    Apply ULP-specific business rules to L/W/H.

    Document AI usually reads the dimensions correctly, but it
    occasionally swaps length and height.

    Business rule:
    Length is very commonly one of:

        48, 72, 74, 79, 96, 98, 120, 144

    If exactly one of the three dimensions matches a preferred
    length value, use that dimension as length.

    Width is generally left as-is unless reordering is required.
    """

    values = [length, width, height]

    # Only work with numeric values.
    numeric_values = [
        v
        for v in values
        if isinstance(v, (int, float))
    ]

    if len(numeric_values) != 3:
        return length, width, height, False

    preferred_matches = [
        v
        for v in values
        if v in PREFERRED_LENGTHS
    ]

    # If exactly one dimension looks like a normal ULP length,
    # use it as length.
    if len(preferred_matches) == 1:

        preferred_length = preferred_matches[0]

        # Already correct.
        if length == preferred_length:
            return length, width, height, False

        # Document AI likely swapped length and height.
        if height == preferred_length:
            return height, width, length, True

        # Less common case:
        # Document AI classified the preferred dimension as width.
        if width == preferred_length:
            return width, length, height, True

    # If more than one dimension matches a preferred length,
    # do not guess.
    return length, width, height, False


def _build_handling_units(
    entities: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Build handling units from repeated dimension / weight / location fields.

    Only use locations on pages where dimensions or weights were
    actually extracted. This avoids false-positive locations from
    printed inventory tables.
    """

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
        e
        for e in entities
        if e.get("page") in measurement_pages
    ]

    lengths = _entities_by_type(
        handling_entities,
        "length",
    )

    widths = _entities_by_type(
        handling_entities,
        "width",
    )

    heights = _entities_by_type(
        handling_entities,
        "height",
    )

    weights = _entities_by_type(
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
                    pages.add(
                        entity.get("page")
                    )

            else:

                unit[field_name] = None
                confidence[field_name] = 0.0

        #
        # Apply ULP-specific dimension normalization.
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

        #
        # If dimensions were rearranged, move the associated
        # confidence values along with them.
        #
        if dimensions_adjusted:

            old_confidence = {
                "length": confidence.get("length", 0.0),
                "width": confidence.get("width", 0.0),
                "height": confidence.get("height", 0.0),
            }

            if normalized_length == original_height:
                new_length_confidence = old_confidence["height"]
            elif normalized_length == original_width:
                new_length_confidence = old_confidence["width"]
            else:
                new_length_confidence = old_confidence["length"]

            if normalized_width == original_length:
                new_width_confidence = old_confidence["length"]
            elif normalized_width == original_height:
                new_width_confidence = old_confidence["height"]
            else:
                new_width_confidence = old_confidence["width"]

            if normalized_height == original_length:
                new_height_confidence = old_confidence["length"]
            elif normalized_height == original_width:
                new_height_confidence = old_confidence["width"]
            else:
                new_height_confidence = old_confidence["height"]

            confidence["length"] = new_length_confidence
            confidence["width"] = new_width_confidence
            confidence["height"] = new_height_confidence

        unit["length"] = normalized_length
        unit["width"] = normalized_width
        unit["height"] = normalized_height

        #
        # Helpful audit field.
        #
        unit["dimensions_adjusted"] = dimensions_adjusted

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
    Convert Document AI's flat entity list into
    Sales Orders with shipment data and handling units.
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
        # Shipment-level fields.
        #
        for field_name in SHIPMENT_FIELDS:

            best = _best_entity(
                order_entities,
                field_name,
            )

            order[
                field_name
            ] = _field_result(
                best
            )

        #
        # Individual handling units / skids.
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
    # Preserve original document order.
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
