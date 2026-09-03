import os
import json
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from pypdf import PdfReader, PdfWriter


MODEL = "gpt-5.4-mini"

# Hard page isolation.
# GPT sees ONE original PDF page at a time.
CHUNK_SIZE = 1


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
# PROMPT
# ==========================================================

EXTRACTION_INSTRUCTIONS = """
You are extracting shipping information from ONE scanned ULP
"Pink" shipping document page.

You are looking at exactly ONE page.

Do not infer or guess information from preceding or following pages.

If a field is not visible on this page, return null.

The most important task is accurately identifying EVERY genuine
handwritten handling-unit / pallet notation on this page.


SALES ORDER

Extract sales_order only if the Sales Order number is visibly
present on THIS PAGE.

Sales Orders commonly begin with:

SO-

If the Sales Order number is not visible:

sales_order = null

Do not guess the Sales Order from a customer PO, consignee,
packing-list number, or other identifier.


SHIPMENT INFORMATION

Extract when visibly present:

- sales_order
- customer_PO
- SRP_number
- delivery_name
- delivery_address
- delivery_address2
- delivery_city
- delivery_state
- delivery_zip
- delivery_contact

Do not move information between fields simply to fill blanks.

Preserve visible wording as accurately as possible.


ADDRESS RULES

delivery_name:
company / institution / destination name.

delivery_address:
primary street address.

delivery_address2:
ATTN line, person name, project name, suite, secondary address
information, or other clearly secondary destination information.

Do not invent address 2.


HANDLING UNITS

Find EVERY genuine handwritten shipping pallet / handling-unit
notation on this page.

Examples may look like:

48 x 45 x 26 / 97# / B8

1 - 74 x 45 x 55 - 542# - D24

48x48x17 / 88# / D19E


For every genuine handling unit extract:

- length
- width
- height
- weight
- location


DIMENSIONS

Dimensions normally appear as:

L x W x H

The first number is normally length.
The second number is normally width.
The third number is normally height.

Common shipment lengths include:

48
72
74
79
96
98
120
144

These common values are contextual clues only.

Do NOT change clearly visible handwriting merely because another
number is more common.


SPECIAL DIMENSION RULE

If handwriting clearly contains two 48 dimensions and one other
dimension, an intended orientation such as:

48 x 48 x other

is common.

Use the visual evidence first.


WEIGHT

The # symbol commonly means pounds.

Examples:

88# = 88 pounds
294# = 294 pounds

Do not confuse quantity numbers with weight.


LOCATION

Location is usually a short handwritten warehouse/staging code.

Examples:

B8
D24
D22E
D19E
C21-2
07
09
A19-E

Do not append unrelated handwritten values to the location.


CRITICAL: PRINTED PRODUCT DIMENSIONS

Do NOT use printed product dimensions, catalog measurements,
item descriptions, quantities, or product-table measurements as
handling-unit dimensions.

Handling-unit information should come from genuine handwritten
shipping notation.


CRITICAL: DO NOT CREATE FAKE HANDLING UNITS

Handwriting alone does not mean a handling unit exists.

Do NOT create a handling-unit record for:

- initials
- signatures
- packed quantities
- miscellaneous notes
- a lone location-looking code
- one isolated number
- marks in a packed/sign column

A genuine handling unit should normally have meaningful evidence
such as:

- multiple dimensions
- dimensions plus weight
- dimensions plus location
- a clearly structured pallet notation


INCOMPLETE HANDLING UNITS

A genuine pallet notation may still have one unreadable value.

Example:

98 x 45 x ? / 294# / C21-2

In that case:

length = 98
width = 45
height = null
weight = 294
location = C21-2
uncertain = true

Do not discard a genuine handling unit just because one component
cannot be read.


UNCERTAINTY

Never invent unreadable values.

Use null for unreadable values.

Set uncertain=true if any part of a genuine handling unit is
ambiguous.

Use notes only to briefly explain the ambiguity.

Do not manufacture numeric confidence percentages.


FINAL CHECK

Before responding:

1. Re-scan the entire page.
2. Confirm whether a Sales Order is visibly present.
3. Count genuine handwritten handling-unit notations.
4. Return every genuine handling unit.
5. Confirm printed product dimensions were not mistaken for pallet
   dimensions.
6. Confirm random handwriting was not turned into a handling unit.
"""


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
    """
    page_index is zero-based.
    """

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
    )


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
# HANDLING UNIT VALIDATION
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


def _handling_unit_has_real_data(
    hu: Dict[str, Any]
) -> bool:
    """
    Keep strong or reasonably incomplete pallet evidence.

    KEEP:

    3 dimensions

    OR

    >= 2 dimensions + weight/location

    OR

    >= 1 dimension + weight + location

    DROP:

    location only
    initials
    random numbers
    packed marks
    weak handwriting
    """

    dimension_count = (
        _count_dimensions(
            hu
        )
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
        and (
            has_weight
            or has_location
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
# ONE PAGE GPT EXTRACTION
# ==========================================================

def _extract_page_with_gpt(
    client,
    page_pdf_bytes: bytes,
    original_page_number: int,
):
    """
    GPT sees exactly one page.

    Python assigns original page number.
    GPT does not decide page numbering.
    """

    uploaded_file_id = None

    try:

        pdf_file = BytesIO(
            page_pdf_bytes
        )

        pdf_file.name = (
            f"ulp_page_"
            f"{original_page_number}.pdf"
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

        response = (
            client.responses.create(
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
                                    EXTRACTION_INSTRUCTIONS,
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
        )

        output_text = (
            response.output_text
            or ""
        ).strip()

        if not output_text:
            raise RuntimeError(
                "OpenAI returned no extraction output."
            )

        try:

            extracted = json.loads(
                output_text
            )

        except json.JSONDecodeError as exc:

            raise RuntimeError(
                "OpenAI returned invalid JSON: "
                f"{str(exc)}"
            )

        page_result = (
            extracted.get(
                "page"
            )
            or {}
        )

        # Python owns page numbering.
        page_result[
            "page"
        ] = original_page_number

        # Add original page to each HU.
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
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
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
# PAGE MATCHING
# ==========================================================

def _field_matches(
    a,
    b
) -> bool:

    a_norm = (
        _normalize_compare_string(
            a
        )
    )

    b_norm = (
        _normalize_compare_string(
            b
        )
    )

    if (
        not a_norm
        or not b_norm
    ):
        return False

    return (
        a_norm == b_norm
    )


def _page_similarity_score(
    a: Dict[str, Any],
    b: Dict[str, Any],
) -> int:
    """
    Score whether two adjacent pages likely belong
    to the same shipment.

    Customer PO is strongest.
    ZIP/name/address provide supporting evidence.
    """

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

    if _field_matches(
        a.get(
            "delivery_contact"
        ),
        b.get(
            "delivery_contact"
        ),
    ):
        score += 2

    return score


# ==========================================================
# SHIPMENT GROUPING
# ==========================================================

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


def _merge_page_into_group(
    group: Dict[str, Any],
    page: Dict[str, Any],
):
    """
    Merge one isolated page into a shipment group.

    Important:
    explicit Sales Order is retained.
    """

    page_number = (
        page.get(
            "page"
        )
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
        _clean_string(
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

    existing_hu_keys = {
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

        if not _handling_unit_has_real_data(
            hu
        ):
            continue

        key = (
            _handling_unit_key(
                hu
            )
        )

        if key in existing_hu_keys:
            continue

        group[
            "handling_units"
        ].append(
            hu
        )

        existing_hu_keys.add(
            key
        )


def _should_start_new_group(
    current_group: Dict[str, Any],
    current_page: Dict[str, Any],
    previous_page: Optional[Dict[str, Any]],
) -> bool:
    """
    Determine whether isolated current_page begins
    a new shipment.

    Rules prioritize explicit Sales Order boundaries.

    This is intentionally conservative.
    """

    if not current_group.get(
        "pages"
    ):
        return False

    current_so = (
        _clean_string(
            current_page.get(
                "sales_order"
            )
        )
    )

    group_so = (
        _clean_string(
            current_group.get(
                "sales_order"
            )
        )
    )

    # ----------------------------------
    # HARD BOUNDARY:
    # new explicit Sales Order differs
    # ----------------------------------

    if (
        current_so
        and group_so
        and current_so.upper()
        != group_so.upper()
    ):
        return True

    # ----------------------------------
    # If current page has explicit SO
    # and existing group has no SO,
    # decide by context.
    # ----------------------------------

    if (
        current_so
        and not group_so
    ):

        if previous_page:

            score = (
                _page_similarity_score(
                    previous_page,
                    current_page,
                )
            )

            if score >= 8:
                return False

        return True

    # ----------------------------------
    # Current page has no SO.
    #
    # Compare to prior page.
    # ----------------------------------

    if not current_so:

        if not previous_page:
            return False

        score = (
            _page_similarity_score(
                previous_page,
                current_page,
            )
        )

        # Strong evidence = continuation.
        if score >= 8:
            return False

        # ----------------------------------
        # Special case:
        # page has essentially no shipment
        # header fields, but may just be a
        # continuation packing list.
        # ----------------------------------

        has_identity_data = any(
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

        if not has_identity_data:
            return False

        # Different recognizable shipment
        # information = new shipment.
        return True

    return False


def _group_pages_into_shipments(
    page_results: List[
        Dict[str, Any]
    ]
) -> List[
    Dict[str, Any]
]:
    """
    Process pages in original document order.

    GPT never joins pages.

    Python creates shipment groups.
    """

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

    # ----------------------------------
    # FINAL CLEANUP
    # ----------------------------------

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
# MAIN EXTRACTOR
# ==========================================================

def extract_ulp_with_gpt(
    pdf_bytes: bytes
):
    """
    ULP GPT extraction.

    Architecture:

    PDF
      -> one page at a time
      -> GPT visually extracts page ONLY
      -> Python assigns exact original page
      -> Python filters weak HUs
      -> Python groups adjacent shipment pages
      -> structured result
    """

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

    client = (
        _get_client()
    )

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

    # ======================================================
    # ONE PAGE PER GPT REQUEST
    # ======================================================

    for page_index in range(
        total_pages
    ):

        original_page_number = (
            page_index + 1
        )

        page_pdf = (
            _make_single_page_pdf(
                pdf_bytes,
                page_index,
            )
        )

        page_result, usage = (
            _extract_page_with_gpt(
                client=client,
                page_pdf_bytes=page_pdf,
                original_page_number=
                    original_page_number,
            )
        )

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
                original_page_number,

            "sales_order":
                page_result.get(
                    "sales_order"
                ),

            "customer_PO":
                page_result.get(
                    "customer_PO"
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
    # PYTHON GROUPS PAGES INTO SHIPMENTS
    # ======================================================

    grouped_shipments = (
        _group_pages_into_shipments(
            page_results
        )
    )

    return {
        "ok":
            True,

        "model":
            MODEL,

        "page_count":
            total_pages,

        "chunk_size":
            1,

        "chunk_count":
            total_pages,

        "extraction": {
            "sales_orders":
                grouped_shipments
        },

        "usage":
            total_usage,

        # Keep for testing.
        # Very useful for seeing exactly what
        # GPT thought existed on each page.
        "pages":
            page_debug,
    }
