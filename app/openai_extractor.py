import os
import re
import json

from io import BytesIO

from typing import (
    Any,
    Dict,
    List,
    Optional,
)

from openai import OpenAI
from pypdf import PdfReader, PdfWriter

from app.document_ai import extract_pink_ocr


MODEL = "gpt-5.4-mini"


# ==========================================================
# BUSINESS RULES
# ==========================================================

SALES_ORDER_PATTERN = re.compile(
    r"^SO-\d{8}$",
    re.IGNORECASE,
)

SALES_ORDER_SEARCH_PATTERN = re.compile(
    r"\bSO-\d{8}\b",
    re.IGNORECASE,
)


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

MIN_TYPICAL_WIDTH = 30
MAX_TYPICAL_WIDTH = 60


# ==========================================================
# OPENAI CLIENT
# ==========================================================

def _get_client():

    api_key = os.getenv(
        "OPENAI_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured."
        )

    return OpenAI(
        api_key=api_key
    )


# ==========================================================
# RESPONSE SCHEMA
# ==========================================================

ULP_SCHEMA = {
    "type": "object",

    "properties": {

        "page": {
            "type": "object",

            "properties": {

                "sales_order": {
                    "type": [
                        "string",
                        "null",
                    ]
                },

                "customer_PO": {
                    "type": [
                        "string",
                        "null",
                    ]
                },

                "SRP_number": {
                    "type": [
                        "string",
                        "null",
                    ]
                },

                "delivery_name": {
                    "type": [
                        "string",
                        "null",
                    ]
                },

                "delivery_address": {
                    "type": [
                        "string",
                        "null",
                    ]
                },

                "delivery_address2": {
                    "type": [
                        "string",
                        "null",
                    ]
                },

                "delivery_city": {
                    "type": [
                        "string",
                        "null",
                    ]
                },

                "delivery_state": {
                    "type": [
                        "string",
                        "null",
                    ]
                },

                "delivery_zip": {
                    "type": [
                        "string",
                        "null",
                    ]
                },

                "delivery_contact": {
                    "type": [
                        "string",
                        "null",
                    ]
                },

                # ==========================================
                # RAW HANDWRITTEN TABLE ROWS
                # ==========================================

                "handling_unit_rows": {
                    "type": "array",

                    "items": {
                        "type": "object",

                        "properties": {

                            "row_index": {
                                "type": "integer"
                            },

                            "raw_dimensions": {
                                "type": [
                                    "string",
                                    "null",
                                ]
                            },

                            "dimensions_are_ditto": {
                                "type": "boolean"
                            },

                            "raw_weight": {
                                "type": [
                                    "string",
                                    "null",
                                ]
                            },

                            "weight_is_ditto": {
                                "type": "boolean"
                            },

                            "raw_location": {
                                "type": [
                                    "string",
                                    "null",
                                ]
                            },

                            "location_is_ditto": {
                                "type": "boolean"
                            },

                            "uncertain": {
                                "type": "boolean"
                            },

                            "notes": {
                                "type": [
                                    "string",
                                    "null",
                                ]
                            },
                        },

                        "required": [
                            "row_index",
                            "raw_dimensions",
                            "dimensions_are_ditto",
                            "raw_weight",
                            "weight_is_ditto",
                            "raw_location",
                            "location_is_ditto",
                            "uncertain",
                            "notes",
                        ],

                        "additionalProperties": False,
                    }
                },
            },

            "required": [
                "sales_order",
                "customer_PO",
                "SRP_number",
                "delivery_name",
                "delivery_address",
                "delivery_address2",
                "delivery_city",
                "delivery_state",
                "delivery_zip",
                "delivery_contact",
                "handling_unit_rows",
            ],

            "additionalProperties": False,
        }
    },

    "required": [
        "page"
    ],

    "additionalProperties": False,
}


# ==========================================================
# GPT INSTRUCTIONS
# ==========================================================

BASE_EXTRACTION_INSTRUCTIONS = """
You are extracting shipping information from exactly ONE ULP
shipping document page.

You receive TWO sources:

1. GOOGLE OCR TEXT
   Use this primarily for PRINTED information.

2. THE ORIGINAL PAGE IMAGE/PDF
   Use this especially for HANDWRITTEN skid / handling-unit rows.

Do not infer information from previous or following PDF pages.


============================================================
SOURCE PRIORITY
============================================================

For printed fields, prefer Google OCR.

For handwritten handling units, visually inspect the original
page image.

Do not trust OCR guesses about handwriting when the page image
shows something different.


============================================================
MASTER SALES ORDER
============================================================

The master ULP Sales Order has this format:

SO-00325428

That is:

SO-
followed by exactly 8 digits.

The master Sales Order is separate from SRP_number.

Examples of SRP values:

SO0316672
S00316672
GT# 6594008 ALTITUDE

NEVER turn an SRP number into a master Sales Order.

Python independently validates the master Sales Order from
Google OCR, so do not invent one.


============================================================
PRINTED SHIPMENT FIELDS
============================================================

Extract:

- customer_PO
- SRP_number
- delivery_name
- delivery_address
- delivery_address2
- delivery_city
- delivery_state
- delivery_zip
- delivery_contact

Preserve printed values faithfully.


============================================================
SHIP-TO ADDRESS
============================================================

delivery_address should be the physical street-address line.

Lines between the organization name and street address may
belong in delivery_address2.

Example:

HOA PLAYGROUND SERVICES LLC
QUEENSLAND MANOR
MEAGAN BIRDSALL
3580 SOUTH 144TH STREET
GILBERT, AZ 85297

becomes:

delivery_name =
HOA PLAYGROUND SERVICES LLC

delivery_address =
3580 SOUTH 144TH STREET

delivery_address2 =
QUEENSLAND MANOR / MEAGAN BIRDSALL

delivery_city =
GILBERT

delivery_state =
AZ

delivery_zip =
85297


============================================================
DELIVERY CONTACT
============================================================

Prefer the field specifically labeled Delivery contact.

A name in the Ship To block is not automatically the
delivery contact.


============================================================
HANDLING-UNIT TABLE — MOST IMPORTANT SECTION
============================================================

Your job is NOT to normalize the handwritten handling units.

Your first job is to TRANSCRIBE THE HANDWRITTEN TABLE ROW BY ROW.

Return EVERY genuine handwritten skid / handling-unit row in
the exact visual TOP-TO-BOTTOM sequence in which it appears.

Use:

row_index = 1
row_index = 2
row_index = 3
...

Do not reorder rows based on dimensions, weight, location,
product, or anything else.

Do not combine identical rows.

Two physically separate skids may have exactly the same:

- dimensions
- weight
- location

They are still TWO handling units and MUST be returned as two
separate rows.


============================================================
RAW DIMENSIONS
============================================================

For an explicitly written dimension set, copy what you see into:

raw_dimensions

Examples:

79 x 32 x 32
98 x 45 x 44
48 x 48 x 17

Do not normalize the numbers yourself.

Python will parse raw_dimensions afterward.


============================================================
RAW WEIGHT
============================================================

Copy the handwritten weight into:

raw_weight

Examples:

162#
756#
195
242#

The # symbol means pounds and may appear before or after the
dimensions.

Do not assume the first handwritten number is length.


============================================================
RAW LOCATION
============================================================

Copy the warehouse/location notation into:

raw_location

Examples:

D19
C17
A23
C26
D21
B21
C-21-2
A19-E


============================================================
DITTO / QUOTATION MARKS — CRITICAL
============================================================

The shipper may use handwritten quotation marks, double quotes,
pairs of short strokes, or ditto marks to mean:

SAME VALUE AS THE IMMEDIATELY PRECEDING HANDLING-UNIT ROW.

Examples of possible marks:

"
''
〃

The mark may be messy and may not look like a perfect
typographic quotation mark.

Interpret the mark according to the COLUMN / POSITION where
it appears.


------------------------------------------------------------
DIMENSION DITTO
------------------------------------------------------------

If the dimension area contains quotation / ditto marks rather
than a new dimension:

dimensions_are_ditto = true
raw_dimensions = null

Do NOT fill in the previous dimensions yourself.

Python will do that later.


------------------------------------------------------------
WEIGHT DITTO
------------------------------------------------------------

If the weight area contains a quotation / ditto mark:

weight_is_ditto = true
raw_weight = null


------------------------------------------------------------
LOCATION DITTO
------------------------------------------------------------

If the location area contains a quotation / ditto mark:

location_is_ditto = true
raw_location = null


------------------------------------------------------------
DITTO MARKS ARE FIELD-SPECIFIC
------------------------------------------------------------

A ditto in dimensions repeats dimensions only.

A ditto in weight repeats weight only.

A ditto in location repeats location only.

Example:

dimension area: "
weight: 167
location: C17

return:

raw_dimensions = null
dimensions_are_ditto = true

raw_weight = "167"
weight_is_ditto = false

raw_location = "C17"
location_is_ditto = false


============================================================
PAGE-5 STYLE EXAMPLE
============================================================

Suppose the handwritten page visually means:

79 x 32 x 32   162   D19
"              162   D19
"              164   D20
"              167   C17
"              165   A23
"              164   C26
79 x 32 x 40   192   D21
"              195   B21

You MUST return EIGHT raw rows in exactly that order.

Row 1:
raw_dimensions = "79 x 32 x 32"
dimensions_are_ditto = false
raw_weight = "162"
raw_location = "D19"

Row 2:
raw_dimensions = null
dimensions_are_ditto = true
raw_weight = "162"
raw_location = "D19"

Row 3:
raw_dimensions = null
dimensions_are_ditto = true
raw_weight = "164"
raw_location = "D20"

Row 4:
raw_dimensions = null
dimensions_are_ditto = true
raw_weight = "167"
raw_location = "C17"

Row 5:
raw_dimensions = null
dimensions_are_ditto = true
raw_weight = "165"
raw_location = "A23"

Row 6:
raw_dimensions = null
dimensions_are_ditto = true
raw_weight = "164"
raw_location = "C26"

Row 7:
raw_dimensions = "79 x 32 x 40"
dimensions_are_ditto = false
raw_weight = "192"
raw_location = "D21"

Row 8:
raw_dimensions = null
dimensions_are_ditto = true
raw_weight = "195"
raw_location = "B21"

Python will later resolve the repeated dimensions.


============================================================
NEW EXPLICIT VALUE RESETS THE DITTO CHAIN
============================================================

If the page says:

79 x 32 x 32
"
"
79 x 32 x 40
"

then the final interpreted sequence is:

79 x 32 x 32
79 x 32 x 32
79 x 32 x 32
79 x 32 x 40
79 x 32 x 40

But YOU should return the RAW rows and the ditto flags.

Do not perform that inheritance yourself.


============================================================
NO BACKWARD INFERENCE
============================================================

A ditto may only refer to the handling-unit row immediately
above it.

Never use a later row to fill an earlier row.

Never invent a value when it is not visible.


============================================================
DO NOT DEDUPLICATE
============================================================

This is extremely important.

If two rows are visually separate but contain identical values,
return both rows.

Example:

79 x 32 x 32 / 162 / D19
79 x 32 x 32 / 162 / D19

must result in TWO rows.

They represent two physical skids.


============================================================
DO NOT CREATE FAKE ROWS
============================================================

Do not create handling-unit rows from:

- signatures
- initials
- random marks
- printed product dimensions
- catalog dimensions
- quantities
- lone numbers unrelated to a skid row

A genuine row may have a missing value.

Preserve it and set uncertain=true if appropriate.


============================================================
FINAL CHECK
============================================================

Before responding:

1. Keep SRP separate from master Sales Order.
2. Use Google OCR for printed information.
3. Use the page image for handwriting.
4. Return every genuine handwritten skid row.
5. Preserve exact top-to-bottom row order.
6. Never deduplicate physically separate rows.
7. Recognize ditto marks by their column position.
8. Set field-specific ditto flags.
9. Do not resolve ditto values yourself.
10. Preserve explicit handwritten weight values.
11. Preserve explicit handwritten locations.
12. Weight may occur before or after dimensions.
13. Do not use printed product dimensions as skid dimensions.
"""


# ==========================================================
# BASIC HELPERS
# ==========================================================

def _clean_string(
    value
) -> Optional[str]:

    if value is None:
        return None

    value = str(
        value
    ).strip()

    return value or None


def _normalize_compare_string(
    value
) -> Optional[str]:

    value = _clean_string(
        value
    )

    if not value:
        return None

    return (
        value
        .upper()
        .replace(".", "")
        .replace(",", "")
        .replace("-", "")
        .replace(" ", "")
        .replace("/", "")
        .replace("\n", "")
    )


def _field_matches(
    a,
    b,
) -> bool:

    a = _normalize_compare_string(
        a
    )

    b = _normalize_compare_string(
        b
    )

    if not a or not b:
        return False

    return a == b


def _first_nonempty(
    current,
    incoming,
):

    if current not in (
        None,
        "",
    ):
        return current

    return incoming


def _append_note(
    hu: Dict[str, Any],
    note: str,
):

    existing = _clean_string(
        hu.get(
            "notes"
        )
    )

    if existing:

        hu[
            "notes"
        ] = (
            existing
            + " "
            + note
        )

    else:

        hu[
            "notes"
        ] = note


# ==========================================================
# MASTER SALES ORDER
# ==========================================================

def _validate_sales_order(
    value
) -> Optional[str]:

    value = _clean_string(
        value
    )

    if not value:
        return None

    value = value.upper()

    if not SALES_ORDER_PATTERN.fullmatch(
        value
    ):
        return None

    return value


def _extract_sales_order_from_ocr(
    page_text: str,
) -> Optional[str]:

    page_text = (
        page_text
        or ""
    )

    if not page_text.strip():
        return None

    lines = [
        line.strip()
        for line in page_text.splitlines()
        if line.strip()
    ]

    for i, line in enumerate(
        lines
    ):

        normalized_label = re.sub(
            r"\s+",
            " ",
            line,
        ).strip().lower()

        compact = (
            normalized_label
            .replace(
                " ",
                ""
            )
        )

        if (
            "sales order"
            not in normalized_label

            and "salesorder"
            not in compact
        ):
            continue

        nearby = "\n".join(
            lines[
                i:min(
                    i + 4,
                    len(lines),
                )
            ]
        )

        match = (
            SALES_ORDER_SEARCH_PATTERN
            .search(
                nearby
            )
        )

        if match:

            return (
                _validate_sales_order(
                    match.group(0)
                )
            )

    matches = (
        SALES_ORDER_SEARCH_PATTERN
        .findall(
            page_text
        )
    )

    unique = []

    seen = set()

    for match in matches:

        value = (
            match.upper()
        )

        if value in seen:
            continue

        seen.add(
            value
        )

        unique.append(
            value
        )

    if len(unique) == 1:

        return _validate_sales_order(
            unique[0]
        )

    return None


# ==========================================================
# PDF HELPERS
# ==========================================================

def _get_page_count(
    pdf_bytes: bytes
) -> int:

    reader = PdfReader(
        BytesIO(
            pdf_bytes
        )
    )

    return len(
        reader.pages
    )


def _make_single_page_pdf(
    pdf_bytes: bytes,
    page_index: int,
) -> bytes:

    reader = PdfReader(
        BytesIO(
            pdf_bytes
        )
    )

    writer = PdfWriter()

    writer.add_page(
        reader.pages[
            page_index
        ]
    )

    output = BytesIO()

    writer.write(
        output
    )

    return output.getvalue()


# ==========================================================
# GOOGLE OCR PAGE MAP
# ==========================================================

def _build_ocr_page_map(
    ocr_result: Dict[str, Any],
) -> Dict[int, str]:

    page_map = {}

    for item in (
        ocr_result.get(
            "pages"
        )
        or []
    ):

        page_number = item.get(
            "page"
        )

        if page_number is None:
            continue

        page_map[
            int(
                page_number
            )
        ] = (
            item.get(
                "text"
            )
            or ""
        )

    return page_map


# ==========================================================
# RAW HU PARSING
# ==========================================================

def _parse_number(
    value: Any,
) -> Optional[float]:

    if value is None:
        return None

    if isinstance(
        value,
        (int, float),
    ):

        return value

    text = str(
        value
    ).strip()

    match = re.search(
        r"\d+(?:\.\d+)?",
        text,
    )

    if not match:
        return None

    number = float(
        match.group(0)
    )

    if number.is_integer():
        return int(
            number
        )

    return number


def _parse_dimensions(
    raw_dimensions: Any,
):

    text = _clean_string(
        raw_dimensions
    )

    if not text:

        return (
            None,
            None,
            None,
        )

    numbers = re.findall(
        r"\d+(?:\.\d+)?",
        text,
    )

    parsed = []

    for value in numbers[:3]:

        number = float(
            value
        )

        if number.is_integer():

            number = int(
                number
            )

        parsed.append(
            number
        )

    while len(parsed) < 3:

        parsed.append(
            None
        )

    return (
        parsed[0],
        parsed[1],
        parsed[2],
    )


def _parse_location(
    raw_location: Any,
) -> Optional[str]:

    return _clean_string(
        raw_location
    )


# ==========================================================
# RESOLVE RAW ROWS + DITTO MARKS
# ==========================================================

def _resolve_handling_unit_rows(
    raw_rows: List[Dict[str, Any]],
    page_number: int,
) -> List[Dict[str, Any]]:
    """
    Convert GPT's visual transcription into actual HUs.

    Rules:
    - preserve every physical row
    - sort by explicit visual row_index
    - resolve ditto marks sequentially
    - new explicit values reset the inherited value
    - inheritance never crosses pages
    """

    rows = [
        row
        for row in raw_rows
        if isinstance(
            row,
            dict,
        )
    ]

    rows.sort(
        key=lambda row: (
            row.get(
                "row_index",
                999999,
            )
        )
    )

    resolved = []

    previous = None

    for raw_row in rows:

        hu = {
            "length":
                None,

            "width":
                None,

            "height":
                None,

            "weight":
                None,

            "location":
                None,

            "uncertain":
                bool(
                    raw_row.get(
                        "uncertain",
                        False,
                    )
                ),

            "notes":
                _clean_string(
                    raw_row.get(
                        "notes"
                    )
                ),

            "page":
                page_number,

            "row_index":
                raw_row.get(
                    "row_index"
                ),
        }

        # --------------------------------------------------
        # DIMENSIONS
        # --------------------------------------------------

        if bool(
            raw_row.get(
                "dimensions_are_ditto",
                False,
            )
        ):

            if previous is not None:

                hu[
                    "length"
                ] = previous.get(
                    "length"
                )

                hu[
                    "width"
                ] = previous.get(
                    "width"
                )

                hu[
                    "height"
                ] = previous.get(
                    "height"
                )

            else:

                hu[
                    "uncertain"
                ] = True

                _append_note(
                    hu,
                    (
                        "Dimension ditto could not be resolved "
                        "because no preceding HU exists on page."
                    ),
                )

        else:

            (
                hu["length"],
                hu["width"],
                hu["height"],
            ) = _parse_dimensions(
                raw_row.get(
                    "raw_dimensions"
                )
            )

        # --------------------------------------------------
        # WEIGHT
        # --------------------------------------------------

        if bool(
            raw_row.get(
                "weight_is_ditto",
                False,
            )
        ):

            if previous is not None:

                hu[
                    "weight"
                ] = previous.get(
                    "weight"
                )

            else:

                hu[
                    "uncertain"
                ] = True

                _append_note(
                    hu,
                    (
                        "Weight ditto could not be resolved "
                        "because no preceding HU exists on page."
                    ),
                )

        else:

            hu[
                "weight"
            ] = _parse_number(
                raw_row.get(
                    "raw_weight"
                )
            )

        # --------------------------------------------------
        # LOCATION
        # --------------------------------------------------

        if bool(
            raw_row.get(
                "location_is_ditto",
                False,
            )
        ):

            if previous is not None:

                hu[
                    "location"
                ] = previous.get(
                    "location"
                )

            else:

                hu[
                    "uncertain"
                ] = True

                _append_note(
                    hu,
                    (
                        "Location ditto could not be resolved "
                        "because no preceding HU exists on page."
                    ),
                )

        else:

            hu[
                "location"
            ] = _parse_location(
                raw_row.get(
                    "raw_location"
                )
            )

        resolved.append(
            hu
        )

        # IMPORTANT:
        # Every resolved physical row becomes the source of
        # inheritance for the next visual row.
        previous = hu

    return resolved


# ==========================================================
# HU PLAUSIBILITY / REPAIR
# ==========================================================

def _count_dimensions(
    hu: Dict[str, Any]
) -> int:

    return sum(
        1
        for field in [
            "length",
            "width",
            "height",
        ]
        if hu.get(
            field
        ) is not None
    )


def _repair_handling_unit(
    hu: Dict[str, Any],
) -> Dict[str, Any]:

    length = hu.get(
        "length"
    )

    width = hu.get(
        "width"
    )

    height = hu.get(
        "height"
    )

    weight = hu.get(
        "weight"
    )

    uncertain = bool(
        hu.get(
            "uncertain"
        )
    )

    # Large height that is probably actually weight.
    if (
        height is not None
        and height >= 240
        and weight is None
        and length is not None
        and width is not None
    ):

        hu[
            "weight"
        ] = height

        hu[
            "height"
        ] = None

        hu[
            "uncertain"
        ] = True

        _append_note(
            hu,
            (
                f"Automatic plausibility correction: "
                f"{height} interpreted as weight rather than height."
            ),
        )

    height = hu.get(
        "height"
    )

    weight = hu.get(
        "weight"
    )

    # Previous ambiguous split protection.
    if (
        uncertain
        and height is not None
        and 60 <= height <= 120
        and weight is not None
        and 0 < weight <= 10
    ):

        hu[
            "height"
        ] = None

        hu[
            "weight"
        ] = None

        hu[
            "uncertain"
        ] = True

        _append_note(
            hu,
            "Possible split handwritten number; height and weight left blank.",
        )

    # Mark extreme dimensions as uncertain.
    for field in [
        "length",
        "width",
        "height",
    ]:

        value = hu.get(
            field
        )

        if (
            value is not None
            and value > 200
        ):

            hu[
                "uncertain"
            ] = True

            _append_note(
                hu,
                (
                    f"{field}={value} is outside normal "
                    f"ULP handling-unit range."
                ),
            )

    height = hu.get(
        "height"
    )

    if (
        height is not None
        and height > 84
    ):

        hu[
            "uncertain"
        ] = True

        _append_note(
            hu,
            (
                f"Height {height} is unusually tall. "
                f"Verify handwritten orientation."
            ),
        )

    return hu


def _partial_dimensions_are_plausible(
    hu: Dict[str, Any],
) -> bool:

    length = hu.get(
        "length"
    )

    width = hu.get(
        "width"
    )

    if (
        length is None
        or width is None
    ):
        return False

    length_ok = (
        length in PREFERRED_LENGTHS
        or 40 <= length <= 160
    )

    width_ok = (
        MIN_TYPICAL_WIDTH
        <= width
        <= MAX_TYPICAL_WIDTH
    )

    return (
        length_ok
        and width_ok
    )


def _handling_unit_has_real_data(
    hu: Dict[str, Any],
) -> bool:

    dimension_count = (
        _count_dimensions(
            hu
        )
    )

    length = hu.get(
        "length"
    )

    width = hu.get(
        "width"
    )

    height = hu.get(
        "height"
    )

    weight = hu.get(
        "weight"
    )

    location = _clean_string(
        hu.get(
            "location"
        )
    )

    has_weight = (
        weight is not None
    )

    has_location = bool(
        location
    )

    dimensional_values = [
        value
        for value in [
            length,
            width,
            height,
        ]
        if value is not None
    ]

    # Great Neck weak-fragment protection.
    if (
        dimension_count <= 2
        and dimensional_values
        and max(
            dimensional_values
        ) < 30
    ):

        return False

    if dimension_count >= 3:

        return True

    if (
        dimension_count >= 2
        and has_weight
        and _partial_dimensions_are_plausible(
            hu
        )
    ):

        return True

    if (
        dimension_count >= 2
        and has_location
        and _partial_dimensions_are_plausible(
            hu
        )
    ):

        return True

    if (
        dimension_count == 1
        and has_weight
        and has_location
    ):

        visible_dimension = next(
            (
                value
                for value in [
                    length,
                    width,
                    height,
                ]
                if value is not None
            ),
            None,
        )

        if (
            visible_dimension is not None
            and visible_dimension >= 40
        ):

            return True

    return False


# ==========================================================
# GPT PAGE EXTRACTION
# ==========================================================

def _extract_page_with_gpt(
    client,
    page_pdf_bytes: bytes,
    original_page_number: int,
    google_ocr_text: str,
):

    uploaded_file_id = None

    try:

        pdf_file = BytesIO(
            page_pdf_bytes
        )

        pdf_file.name = (
            f"ulp_page_{original_page_number}.pdf"
        )

        uploaded_file = (
            client.files.create(
                file=pdf_file,
                purpose="user_data",
            )
        )

        uploaded_file_id = (
            uploaded_file.id
        )

        page_prompt = (
            BASE_EXTRACTION_INSTRUCTIONS
            + "\n\n"
            + "============================================================\n"
            + "GOOGLE OCR TEXT FOR THIS PAGE\n"
            + "============================================================\n"
            + "\n"
            + (
                google_ocr_text.strip()
                if google_ocr_text.strip()
                else "[NO GOOGLE OCR TEXT RETURNED]"
            )
        )

        response = client.responses.create(

            model=MODEL,

            input=[
                {
                    "role": "user",

                    "content": [

                        {
                            "type": "input_text",
                            "text": page_prompt,
                        },

                        {
                            "type": "input_file",
                            "file_id": uploaded_file_id,
                        },
                    ],
                }
            ],

            text={
                "format": {

                    "type":
                        "json_schema",

                    "name":
                        "ulp_pink_page_extraction",

                    "strict":
                        True,

                    "schema":
                        ULP_SCHEMA,
                }
            },
        )

        output_text = (
            response.output_text
            or ""
        ).strip()

        if not output_text:

            raise RuntimeError(
                "OpenAI returned no extraction output."
            )

        extracted = json.loads(
            output_text
        )

        page_result = (
            extracted.get(
                "page"
            )
            or {}
        )

        page_result[
            "sales_order"
        ] = _validate_sales_order(
            page_result.get(
                "sales_order"
            )
        )

        page_result[
            "page"
        ] = original_page_number

        # ==================================================
        # RAW ROWS -> RESOLVED HUs
        # ==================================================

        raw_rows = (
            page_result.get(
                "handling_unit_rows"
            )
            or []
        )

        resolved_hus = (
            _resolve_handling_unit_rows(
                raw_rows,
                original_page_number,
            )
        )

        cleaned_hus = []

        for hu in resolved_hus:

            hu = (
                _repair_handling_unit(
                    hu
                )
            )

            if _handling_unit_has_real_data(
                hu
            ):

                cleaned_hus.append(
                    hu
                )

        # Downstream contract stays unchanged.
        page_result[
            "handling_units"
        ] = cleaned_hus

        # Keep raw transcription in debug page result.
        page_result[
            "raw_handling_unit_rows"
        ] = raw_rows

        # Remove schema-only original key.
        page_result.pop(
            "handling_unit_rows",
            None,
        )

        # ==================================================
        # USAGE
        # ==================================================

        usage = {
            "input_tokens":
                0,

            "output_tokens":
                0,

            "total_tokens":
                0,
        }

        if response.usage:

            usage[
                "input_tokens"
            ] = (
                response.usage.input_tokens
                or 0
            )

            usage[
                "output_tokens"
            ] = (
                response.usage.output_tokens
                or 0
            )

            usage[
                "total_tokens"
            ] = (
                response.usage.total_tokens
                or 0
            )

        return (
            page_result,
            usage,
        )

    finally:

        if uploaded_file_id:

            try:

                client.files.delete(
                    uploaded_file_id
                )

            except Exception:
                pass


# ==========================================================
# PAGE SIMILARITY
# ==========================================================

def _page_similarity_score(
    a: Dict[str, Any],
    b: Dict[str, Any],
) -> int:

    score = 0

    if _field_matches(
        a.get(
            "customer_PO"
        ),
        b.get(
            "customer_PO"
        ),
    ):
        score += 12

    if _field_matches(
        a.get(
            "delivery_zip"
        ),
        b.get(
            "delivery_zip"
        ),
    ):
        score += 6

    if _field_matches(
        a.get(
            "delivery_name"
        ),
        b.get(
            "delivery_name"
        ),
    ):
        score += 5

    if _field_matches(
        a.get(
            "delivery_address"
        ),
        b.get(
            "delivery_address"
        ),
    ):
        score += 5

    if _field_matches(
        a.get(
            "delivery_city"
        ),
        b.get(
            "delivery_city"
        ),
    ):
        score += 2

    if _field_matches(
        a.get(
            "delivery_state"
        ),
        b.get(
            "delivery_state"
        ),
    ):
        score += 1

    return score


# ==========================================================
# SHIPMENT GROUPING
# ==========================================================

SHIPMENT_FIELDS = [
    "customer_PO",
    "SRP_number",
    "delivery_name",
    "delivery_address",
    "delivery_address2",
    "delivery_city",
    "delivery_state",
    "delivery_zip",
    "delivery_contact",
]


def _new_shipment_group():

    return {
        "sales_order":
            None,

        "pages":
            [],

        "customer_PO":
            None,

        "SRP_number":
            None,

        "delivery_name":
            None,

        "delivery_address":
            None,

        "delivery_address2":
            None,

        "delivery_city":
            None,

        "delivery_state":
            None,

        "delivery_zip":
            None,

        "delivery_contact":
            None,

        "handling_units":
            [],
    }


def _merge_page_into_group(
    group: Dict[str, Any],
    page: Dict[str, Any],
):
    """
    IMPORTANT:
    We intentionally DO NOT deduplicate handling units.

    Two separate physical skids are allowed to have identical
    dimensions, weight and location.
    """

    page_number = page.get(
        "page"
    )

    if (
        page_number is not None
        and page_number not in group[
            "pages"
        ]
    ):

        group[
            "pages"
        ].append(
            page_number
        )

    page_so = (
        _validate_sales_order(
            page.get(
                "sales_order"
            )
        )
    )

    if (
        not group.get(
            "sales_order"
        )
        and page_so
    ):

        group[
            "sales_order"
        ] = page_so

    for field in SHIPMENT_FIELDS:

        group[
            field
        ] = _first_nonempty(
            group.get(
                field
            ),
            page.get(
                field
            ),
        )

    for hu in (
        page.get(
            "handling_units"
        )
        or []
    ):

        group[
            "handling_units"
        ].append(
            hu
        )


# ==========================================================
# GROUP BOUNDARIES
# ==========================================================

def _should_start_new_group(
    current_group: Dict[str, Any],
    current_page: Dict[str, Any],
    previous_page: Optional[
        Dict[str, Any]
    ],
) -> bool:

    if not current_group[
        "pages"
    ]:

        return False

    current_so = (
        _validate_sales_order(
            current_page.get(
                "sales_order"
            )
        )
    )

    group_so = (
        _validate_sales_order(
            current_group.get(
                "sales_order"
            )
        )
    )

    if (
        current_so
        and group_so
        and current_so != group_so
    ):

        return True

    if not previous_page:

        return False

    similarity = (
        _page_similarity_score(
            previous_page,
            current_page,
        )
    )

    if similarity >= 8:

        return False

    if (
        current_so
        and not group_so
    ):

        return True

    has_identity = any(
        _clean_string(
            current_page.get(
                field
            )
        )

        for field in [
            "customer_PO",
            "delivery_name",
            "delivery_address",
            "delivery_zip",
        ]
    )

    if not has_identity:

        return False

    return True


# ==========================================================
# BACKWARD STITCHING
# ==========================================================

def _group_has_identity(
    group: Dict[str, Any],
) -> bool:

    return any(
        _clean_string(
            group.get(
                field
            )
        )

        for field in [
            "customer_PO",
            "delivery_name",
            "delivery_address",
            "delivery_zip",
        ]
    )


def _group_is_so_only_or_packing_list(
    group: Dict[str, Any],
) -> bool:

    if not _validate_sales_order(
        group.get(
            "sales_order"
        )
    ):
        return False

    identity_count = sum(
        1

        for field in [
            "customer_PO",
            "delivery_name",
            "delivery_address",
            "delivery_zip",
        ]

        if _clean_string(
            group.get(
                field
            )
        )
    )

    return (
        identity_count <= 1
    )


def _groups_are_adjacent(
    left: Dict[str, Any],
    right: Dict[str, Any],
) -> bool:

    if (
        not left.get(
            "pages"
        )
        or not right.get(
            "pages"
        )
    ):
        return False

    return (
        max(
            left[
                "pages"
            ]
        )
        + 1
        ==
        min(
            right[
                "pages"
            ]
        )
    )


def _merge_group_into_group(
    target: Dict[str, Any],
    source: Dict[str, Any],
):
    """
    Also intentionally preserves duplicate physical HUs.
    """

    for page in (
        source.get(
            "pages"
        )
        or []
    ):

        if page not in target[
            "pages"
        ]:

            target[
                "pages"
            ].append(
                page
            )

    source_so = (
        _validate_sales_order(
            source.get(
                "sales_order"
            )
        )
    )

    if (
        not target.get(
            "sales_order"
        )
        and source_so
    ):

        target[
            "sales_order"
        ] = source_so

    for field in SHIPMENT_FIELDS:

        target[
            field
        ] = _first_nonempty(
            target.get(
                field
            ),
            source.get(
                field
            ),
        )

    for hu in (
        source.get(
            "handling_units"
        )
        or []
    ):

        target[
            "handling_units"
        ].append(
            hu
        )


def _backward_stitch_sales_orders(
    groups: List[
        Dict[str, Any]
    ],
) -> List[
    Dict[str, Any]
]:

    if len(
        groups
    ) < 2:

        return groups

    stitched = []

    i = 0

    while i < len(
        groups
    ):

        current = groups[
            i
        ]

        if (
            i + 1
            < len(
                groups
            )
        ):

            nxt = groups[
                i + 1
            ]

            should_stitch = (

                not _validate_sales_order(
                    current.get(
                        "sales_order"
                    )
                )

                and _group_has_identity(
                    current
                )

                and _group_is_so_only_or_packing_list(
                    nxt
                )

                and _groups_are_adjacent(
                    current,
                    nxt,
                )
            )

            if should_stitch:

                _merge_group_into_group(
                    current,
                    nxt,
                )

                stitched.append(
                    current
                )

                i += 2

                continue

        stitched.append(
            current
        )

        i += 1

    return stitched


# ==========================================================
# GROUP PAGES INTO SHIPMENTS
# ==========================================================

def _group_pages_into_shipments(
    page_results: List[
        Dict[str, Any]
    ],
) -> List[
    Dict[str, Any]
]:

    groups = []

    current_group = (
        _new_shipment_group()
    )

    previous_page = None

    for page in page_results:

        if _should_start_new_group(
            current_group,
            page,
            previous_page,
        ):

            if current_group[
                "pages"
            ]:

                groups.append(
                    current_group
                )

            current_group = (
                _new_shipment_group()
            )

        _merge_page_into_group(
            current_group,
            page,
        )

        previous_page = page

    if current_group[
        "pages"
    ]:

        groups.append(
            current_group
        )

    groups = (
        _backward_stitch_sales_orders(
            groups
        )
    )

    for group in groups:

        group[
            "pages"
        ] = sorted(
            set(
                group[
                    "pages"
                ]
            )
        )

        # Preserve page first, then physical row order.
        group[
            "handling_units"
        ].sort(
            key=lambda hu: (
                hu.get(
                    "page",
                    999999,
                ),
                hu.get(
                    "row_index",
                    999999,
                ),
            )
        )

    return groups


# ==========================================================
# MAIN HYBRID EXTRACTOR
# ==========================================================

def extract_ulp_with_gpt(
    pdf_bytes: bytes,
):

    if not pdf_bytes:

        raise ValueError(
            "PDF is empty."
        )

    total_pages = (
        _get_page_count(
            pdf_bytes
        )
    )

    if total_pages <= 0:

        raise ValueError(
            "PDF contains no pages."
        )

    # ======================================================
    # GOOGLE ENTERPRISE OCR
    # ======================================================

    ocr_result = (
        extract_pink_ocr(
            pdf_bytes
        )
    )

    ocr_page_map = (
        _build_ocr_page_map(
            ocr_result
        )
    )

    # ======================================================
    # GPT PAGE-BY-PAGE
    # ======================================================

    client = _get_client()

    page_results = []

    page_debug = []

    total_usage = {
        "input_tokens":
            0,

        "output_tokens":
            0,

        "total_tokens":
            0,
    }

    ocr_sales_orders_found = 0

    for page_index in range(
        total_pages
    ):

        page_number = (
            page_index + 1
        )

        page_pdf = (
            _make_single_page_pdf(
                pdf_bytes,
                page_index,
            )
        )

        google_ocr_text = (
            ocr_page_map.get(
                page_number,
                "",
            )
        )

        google_sales_order = (
            _extract_sales_order_from_ocr(
                google_ocr_text
            )
        )

        if google_sales_order:

            ocr_sales_orders_found += 1

        (
            page_result,
            usage,
        ) = _extract_page_with_gpt(

            client=
                client,

            page_pdf_bytes=
                page_pdf,

            original_page_number=
                page_number,

            google_ocr_text=
                google_ocr_text,
        )

        gpt_sales_order = (
            _validate_sales_order(
                page_result.get(
                    "sales_order"
                )
            )
        )

        final_sales_order = (
            google_sales_order
            or gpt_sales_order
        )

        page_result[
            "sales_order"
        ] = final_sales_order

        page_results.append(
            page_result
        )

        total_usage[
            "input_tokens"
        ] += usage[
            "input_tokens"
        ]

        total_usage[
            "output_tokens"
        ] += usage[
            "output_tokens"
        ]

        total_usage[
            "total_tokens"
        ] += usage[
            "total_tokens"
        ]

        page_debug.append({

            "page":
                page_number,

            "sales_order":
                final_sales_order,

            "google_sales_order":
                google_sales_order,

            "gpt_sales_order":
                gpt_sales_order,

            "customer_PO":
                page_result.get(
                    "customer_PO"
                ),

            "SRP_number":
                page_result.get(
                    "SRP_number"
                ),

            "delivery_name":
                page_result.get(
                    "delivery_name"
                ),

            "delivery_address":
                page_result.get(
                    "delivery_address"
                ),

            "delivery_address2":
                page_result.get(
                    "delivery_address2"
                ),

            "delivery_city":
                page_result.get(
                    "delivery_city"
                ),

            "delivery_state":
                page_result.get(
                    "delivery_state"
                ),

            "delivery_zip":
                page_result.get(
                    "delivery_zip"
                ),

            "delivery_contact":
                page_result.get(
                    "delivery_contact"
                ),

            # Critical debugging for this revision.
            "raw_handling_unit_rows":
                page_result.get(
                    "raw_handling_unit_rows",
                    []
                ),

            "handling_units":
                page_result.get(
                    "handling_units",
                    []
                ),

            "handling_unit_count":
                len(
                    page_result.get(
                        "handling_units"
                    )
                    or []
                ),

            "ocr_text_length":
                len(
                    google_ocr_text
                ),

            "usage":
                usage,
        })

    grouped_shipments = (
        _group_pages_into_shipments(
            page_results
        )
    )

    return {

        "ok":
            True,

        "mode":
            "google_ocr_plus_gpt_raw_hu_rows",

        "model":
            MODEL,

        "page_count":
            total_pages,

        "chunk_size":
            1,

        "chunk_count":
            total_pages,

        "google_ocr": {

            "processor":
                "711477af4e3c321d",

            "page_count":
                ocr_result.get(
                    "page_count"
                ),

            "chunk_count":
                ocr_result.get(
                    "chunk_count"
                ),

            "sales_orders_found":
                ocr_sales_orders_found,
        },

        "extraction": {
            "sales_orders":
                grouped_shipments
        },

        "usage":
            total_usage,

        "pages":
            page_debug,
    }
