import os
import re
import json
from io import BytesIO
from typing import Any, Dict, List, Optional

from openai import OpenAI
from pypdf import PdfReader, PdfWriter

from app.document_ai import extract_pink_ocr


MODEL = "gpt-5.4-mini"
CHUNK_SIZE = 1


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
                    "type": ["string", "null"]
                },

                "customer_PO": {
                    "type": ["string", "null"]
                },

                "SRP_number": {
                    "type": ["string", "null"]
                },

                "delivery_name": {
                    "type": ["string", "null"]
                },

                "delivery_address": {
                    "type": ["string", "null"]
                },

                "delivery_address2": {
                    "type": ["string", "null"]
                },

                "delivery_city": {
                    "type": ["string", "null"]
                },

                "delivery_state": {
                    "type": ["string", "null"]
                },

                "delivery_zip": {
                    "type": ["string", "null"]
                },

                "delivery_contact": {
                    "type": ["string", "null"]
                },

                "handling_units": {
                    "type": "array",

                    "items": {
                        "type": "object",

                        "properties": {

                            "length": {
                                "type": ["number", "null"]
                            },

                            "width": {
                                "type": ["number", "null"]
                            },

                            "height": {
                                "type": ["number", "null"]
                            },

                            "weight": {
                                "type": ["number", "null"]
                            },

                            "location": {
                                "type": ["string", "null"]
                            },

                            "uncertain": {
                                "type": "boolean"
                            },

                            "notes": {
                                "type": ["string", "null"]
                            }
                        },

                        "required": [
                            "length",
                            "width",
                            "height",
                            "weight",
                            "location",
                            "uncertain",
                            "notes"
                        ],

                        "additionalProperties": False
                    }
                }
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
                "handling_units"
            ],

            "additionalProperties": False
        }
    },

    "required": [
        "page"
    ],

    "additionalProperties": False
}


# ==========================================================
# GPT BASE INSTRUCTIONS
# ==========================================================

BASE_EXTRACTION_INSTRUCTIONS = """
You are extracting shipping information from exactly ONE ULP
shipping document page.

You receive TWO sources:

1. GOOGLE OCR TEXT
   This is intended primarily for PRINTED information.

2. THE ORIGINAL PAGE IMAGE/PDF
   Use this especially for HANDWRITTEN handling-unit information.

Do not infer information from previous or following pages.


============================================================
SOURCE PRIORITY
============================================================

For printed fields, use the Google OCR text whenever possible.

For handwriting, visually inspect the original page.

Do not use Google OCR guesses about handwriting when the page
image shows something different.


============================================================
MASTER SALES ORDER
============================================================

The master ULP Sales Order normally looks like:

SO-00325428

Format:

SO-
followed by exactly 8 digits.

The master Sales Order is completely separate from SRP_number.

Examples of SRP numbers:

SO0316672
S00316672

NEVER convert an SRP number into a master Sales Order.

Do not add a hyphen to an SRP number.

Python will independently validate and recover the master
Sales Order from Google OCR, so do not invent one.


============================================================
PRINTED SHIPMENT FIELDS
============================================================

Use the provided Google OCR text plus the page layout to extract:

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
SHIP-TO / STREET ADDRESS
============================================================

Read the entire Ship To block.

delivery_address should be the physical street-address line.

Street addresses often look like:

16777 FILLMORE ST
3580 SOUTH 144TH STREET
401 N FAIRVIEW AVE
499 W ELM STREET
1135 NM 554

A street address normally begins with a building/street number
followed by a street, highway, road, route, etc.

Lines between the organization name and physical street address
may belong in delivery_address2.

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

Prefer the printed field specifically labeled Delivery contact.

A person's name in the Ship To block is not automatically the
delivery contact.


============================================================
HANDLING UNITS — PRIMARY GPT JOB
============================================================

Visually inspect the original page for EVERY genuine handwritten
handling-unit / pallet notation.

Examples:

48 x 45 x 26 / 97# / B8

74 x 45 x 55 / 542# / D24

48 x 48 x 17 / 88# / D19E

98 x 45 x ? / 294# / C21-2


Extract:

- length
- width
- height
- weight
- location


============================================================
DIMENSION / WEIGHT RULES
============================================================

Dimensions normally appear:

L x W x H

Typical lengths include:

48
72
74
79
96
98
120
144

Typical widths include:

40
42
44
45
48

These are clues only.

Do not force a common dimension if the handwriting says
something else.

The # symbol normally means pounds.

Example:

756# = 756 pounds


Do not interpret:

48 x 44 / 756# / C27

as:

48 x 44 x 756

Correct interpretation is approximately:

length = 48
width = 44
height = null
weight = 756
location = C27


============================================================
AMBIGUOUS HANDWRITING
============================================================

Be careful with:

19 vs 91
17 vs 71
14 vs 41
78 vs 98

If genuinely ambiguous:

- return the most likely visible value
- set uncertain=true
- explain briefly in notes

Do not make up a missing value.


============================================================
DO NOT CREATE FAKE HANDLING UNITS
============================================================

Do not create HUs from:

- signatures
- initials
- check marks
- isolated numbers
- lone warehouse locations
- printed product measurements
- catalog measurements
- quantities


A real HU may still have one missing field.

Example:

98 x 45 x ? / 294# / C21-2

is a real HU:

length = 98
width = 45
height = null
weight = 294
location = C21-2
uncertain = true


============================================================
FINAL CHECK
============================================================

Before responding:

1. Keep SRP separate from master Sales Order.
2. Use Google OCR for printed fields.
3. Use the page image for handwriting.
4. Identify the actual physical street address.
5. Preserve relevant Address 2 lines.
6. Find every real handwritten HU.
7. Do not confuse weight with height.
8. Do not use printed product dimensions as HU dimensions.
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
    b
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
    incoming
):

    if current not in (
        None,
        "",
    ):
        return current

    return incoming


# ==========================================================
# MASTER SALES ORDER VALIDATION
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


# ==========================================================
# GOOGLE OCR SALES ORDER EXTRACTION
# ==========================================================

def _extract_sales_order_from_ocr(
    page_text: str,
) -> Optional[str]:
    """
    Deterministically recover the master SO from
    Google Enterprise OCR.

    Priority:

    1. Find the literal Sales order label and search
       its line / nearby lines.

    2. If there is exactly ONE SO-######## anywhere
       on the page, accept it.

    We NEVER transform SRP values such as:
        SO0316672
        S00316672
    """

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

    # ------------------------------------------------------
    # FIRST:
    # Look specifically around a Sales order label.
    # ------------------------------------------------------

    for i, line in enumerate(
        lines
    ):

        normalized_label = re.sub(
            r"\s+",
            " ",
            line,
        ).strip().lower()

        if (
            "sales order" not in normalized_label
            and "salesorder" not in normalized_label.replace(
                " ",
                ""
            )
        ):
            continue

        # Search the label line plus the next few OCR lines.
        nearby_lines = lines[
            i:min(
                i + 4,
                len(lines)
            )
        ]

        nearby_text = "\n".join(
            nearby_lines
        )

        match = SALES_ORDER_SEARCH_PATTERN.search(
            nearby_text
        )

        if match:

            return _validate_sales_order(
                match.group(0)
            )

    # ------------------------------------------------------
    # FALLBACK:
    #
    # If Google sees exactly one valid SO-######## anywhere
    # on this page, it is safe to use.
    #
    # This does NOT match SO0316672 because the hyphen is
    # required.
    # ------------------------------------------------------

    matches = SALES_ORDER_SEARCH_PATTERN.findall(
        page_text
    )

    unique_matches = []

    seen = set()

    for match in matches:

        normalized = match.upper()

        if normalized not in seen:

            seen.add(
                normalized
            )

            unique_matches.append(
                normalized
            )

    if len(
        unique_matches
    ) == 1:

        return _validate_sales_order(
            unique_matches[0]
        )

    return None


# ==========================================================
# PDF HELPERS
# ==========================================================

def _get_page_count(
    pdf_bytes: bytes
) -> int:

    reader = PdfReader(
        BytesIO(pdf_bytes)
    )

    return len(
        reader.pages
    )


def _make_single_page_pdf(
    pdf_bytes: bytes,
    page_index: int,
) -> bytes:

    reader = PdfReader(
        BytesIO(pdf_bytes)
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
            int(page_number)
        ] = (
            item.get(
                "text"
            )
            or ""
        )

    return page_map


# ==========================================================
# HANDLING UNIT HELPERS
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


def _repair_handling_unit(
    hu: Dict[str, Any]
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

    # ------------------------------------------------------
    # LARGE HEIGHT THAT IS REALLY LIKELY A WEIGHT
    # ------------------------------------------------------

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
                f"{height} was extracted as height but "
                f"is more plausible as weight. "
                f"Height left blank for review."
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

    # ------------------------------------------------------
    # POSSIBLE SPLIT NUMBER
    #
    # 756 -> 75 + 6
    # ------------------------------------------------------

    if (
        uncertain
        and height is not None
        and 60 <= height <= 120
        and weight is not None
        and 0 < weight <= 10
    ):

        bad_height = height
        bad_weight = weight

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
            (
                f"Possible handwritten number split: "
                f"height={bad_height}, weight={bad_weight}. "
                f"Both values left blank for review."
            )
        )

    # ------------------------------------------------------
    # CONTRADICTORY / QUESTIONABLE HEIGHT
    # ------------------------------------------------------

    height = hu.get(
        "height"
    )

    weight = hu.get(
        "weight"
    )

    if (
        bool(
            hu.get(
                "uncertain"
            )
        )
        and height is not None
        and 60 <= height <= 120
        and weight is not None
        and weight >= 100
    ):

        note_text = (
            _clean_string(
                hu.get(
                    "notes"
                )
            )
            or ""
        ).lower()

        height_is_described_unclear = (

            "height is not clearly" in note_text

            or "height is unclear" in note_text

            or "height is not legible" in note_text

            or "height is unreadable" in note_text

            or "height is not visible" in note_text

            or "height is somewhat ambiguous" in note_text

            or (
                "height" in note_text
                and "ambiguous" in note_text
            )

            or (
                "height" in note_text
                and "unclear" in note_text
            )

            or (
                "height" in note_text
                and "not visible" in note_text
            )
        )

        if height_is_described_unclear:

            questionable_height = height

            hu[
                "height"
            ] = None

            hu[
                "uncertain"
            ] = True

            _append_note(
                hu,
                (
                    f"Height {questionable_height} was removed "
                    f"because the visual extraction identified "
                    f"the height as ambiguous while weight "
                    f"{weight} was independently identified."
                )
            )

    # ------------------------------------------------------
    # EXTREME DIMENSIONS
    # ------------------------------------------------------

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
                    f"{field}={value} is outside "
                    f"normal ULP handling-unit range."
                )
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
            )
        )

    return hu


def _partial_dimensions_are_plausible(
    hu: Dict[str, Any]
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
    hu: Dict[str, Any]
) -> bool:

    dimension_count = _count_dimensions(
        hu
    )

    has_weight = (
        hu.get(
            "weight"
        ) is not None
    )

    has_location = bool(
        _clean_string(
            hu.get(
                "location"
            )
        )
    )

    if dimension_count >= 3:
        return True

    if (
        dimension_count >= 2
        and has_weight
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
        dimension_count >= 1
        and has_weight
        and has_location
    ):
        return True

    return False


def _handling_unit_key(
    hu: Dict[str, Any]
):

    return (
        hu.get(
            "page"
        ),

        hu.get(
            "length"
        ),

        hu.get(
            "width"
        ),

        hu.get(
            "height"
        ),

        hu.get(
            "weight"
        ),

        _normalize_compare_string(
            hu.get(
                "location"
            )
        ),
    )


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
            f"ulp_page_"
            f"{original_page_number}.pdf"
        )

        uploaded_file = client.files.create(
            file=pdf_file,
            purpose="user_data",
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
                    "role":
                        "user",

                    "content": [

                        {
                            "type":
                                "input_text",

                            "text":
                                page_prompt,
                        },

                        {
                            "type":
                                "input_file",

                            "file_id":
                                uploaded_file_id,
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

        cleaned_hus = []

        for hu in (
            page_result.get(
                "handling_units"
            )
            or []
        ):

            hu[
                "page"
            ] = original_page_number

            hu = _repair_handling_unit(
                hu
            )

            if _handling_unit_has_real_data(
                hu
            ):

                cleaned_hus.append(
                    hu
                )

        page_result[
            "handling_units"
        ] = cleaned_hus

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
# SHIPMENT GROUPS
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

    page_so = _validate_sales_order(
        page.get(
            "sales_order"
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

    existing_keys = {
        _handling_unit_key(
            hu
        )
        for hu in group[
            "handling_units"
        ]
    }

    for hu in (
        page.get(
            "handling_units"
        )
        or []
    ):

        key = _handling_unit_key(
            hu
        )

        if key in existing_keys:
            continue

        group[
            "handling_units"
        ].append(
            hu
        )

        existing_keys.add(
            key
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

    current_so = _validate_sales_order(
        current_page.get(
            "sales_order"
        )
    )

    group_so = _validate_sales_order(
        current_group.get(
            "sales_order"
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

    similarity = _page_similarity_score(
        previous_page,
        current_page,
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
# BACKWARD SALES ORDER STITCHING
# ==========================================================

def _group_has_identity(
    group: Dict[str, Any]
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
    group: Dict[str, Any]
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

    source_so = _validate_sales_order(
        source.get(
            "sales_order"
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

    existing_keys = {
        _handling_unit_key(
            hu
        )
        for hu in target[
            "handling_units"
        ]
    }

    for hu in (
        source.get(
            "handling_units"
        )
        or []
    ):

        key = _handling_unit_key(
            hu
        )

        if key in existing_keys:
            continue

        target[
            "handling_units"
        ].append(
            hu
        )

        existing_keys.add(
            key
        )


def _backward_stitch_sales_orders(
    groups: List[
        Dict[str, Any]
    ]
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
    ]
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

    groups = _backward_stitch_sales_orders(
        groups
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

        group[
            "handling_units"
        ].sort(
            key=lambda hu: (
                hu.get(
                    "page"
                )
                if hu.get(
                    "page"
                ) is not None
                else 999999
            )
        )

    return groups


# ==========================================================
# MAIN HYBRID EXTRACTOR
# ==========================================================

def extract_ulp_with_gpt(
    pdf_bytes: bytes
):

    if not pdf_bytes:

        raise ValueError(
            "PDF is empty."
        )

    total_pages = _get_page_count(
        pdf_bytes
    )

    if total_pages <= 0:

        raise ValueError(
            "PDF contains no pages."
        )

    # ======================================================
    # STEP 1
    #
    # GOOGLE ENTERPRISE OCR — ONCE FOR THE WHOLE PDF
    # ======================================================

    ocr_result = extract_pink_ocr(
        pdf_bytes
    )

    ocr_page_map = _build_ocr_page_map(
        ocr_result
    )

    # ======================================================
    # STEP 2
    #
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

        page_pdf = _make_single_page_pdf(
            pdf_bytes,
            page_index,
        )

        google_ocr_text = (
            ocr_page_map.get(
                page_number,
                "",
            )
        )

        # --------------------------------------------------
        # DETERMINISTIC GOOGLE MASTER SO
        # --------------------------------------------------

        google_sales_order = (
            _extract_sales_order_from_ocr(
                google_ocr_text
            )
        )

        if google_sales_order:

            ocr_sales_orders_found += 1

        # --------------------------------------------------
        # GPT
        # --------------------------------------------------

        page_result, usage = (
            _extract_page_with_gpt(

                client=
                    client,

                page_pdf_bytes=
                    page_pdf,

                original_page_number=
                    page_number,

                google_ocr_text=
                    google_ocr_text,
            )
        )

        gpt_sales_order = (
            _validate_sales_order(
                page_result.get(
                    "sales_order"
                )
            )
        )

        # --------------------------------------------------
        # MASTER SALES ORDER SOURCE PRIORITY
        #
        # 1. GOOGLE PRINTED OCR
        # 2. GPT ONLY IF GOOGLE FOUND NOTHING
        #
        # SRP IS NEVER A FALLBACK.
        # --------------------------------------------------

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

        # --------------------------------------------------
        # DEBUG
        # --------------------------------------------------

        page_debug.append({

            "page":
                page_number,

            "sales_order":
                final_sales_order,

            "google_sales_order":
                google_sales_order,

            "gpt_sales_order":
                gpt_sales_order,

            "SRP_number":
                page_result.get(
                    "SRP_number"
                ),

            "customer_PO":
                page_result.get(
                    "customer_PO"
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

            "ocr_text_length":
                len(
                    google_ocr_text
                ),

            "handling_unit_count":
                len(
                    page_result.get(
                        "handling_units"
                    )
                    or []
                ),

            "usage":
                usage,
        })

    # ======================================================
    # STEP 3
    #
    # GROUP PAGES INTO SHIPMENTS
    # ======================================================

    grouped_shipments = (
        _group_pages_into_shipments(
            page_results
        )
    )

    return {

        "ok":
            True,

        "mode":
            "google_ocr_plus_gpt",

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
