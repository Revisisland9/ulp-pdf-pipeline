import os
import json
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from pypdf import PdfReader, PdfWriter


MODEL = "gpt-5.4-mini"
CHUNK_SIZE = 3


# ==========================================================
# OPENAI CLIENT
# ==========================================================

def _get_client():
    api_key = os.getenv("OPENAI_API_KEY")

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
        "sales_orders": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "sales_order": {
                        "type": ["string", "null"]
                    },

                    "pages": {
                        "type": "array",
                        "items": {
                            "type": "integer"
                        }
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

                                "page": {
                                    "type": ["integer", "null"]
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
                                "page",
                                "uncertain",
                                "notes"
                            ],

                            "additionalProperties": False
                        }
                    }
                },

                "required": [
                    "sales_order",
                    "pages",
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
        }
    },

    "required": [
        "sales_orders"
    ],

    "additionalProperties": False
}


# ==========================================================
# PROMPT
# ==========================================================

BASE_EXTRACTION_INSTRUCTIONS = """
You are extracting shipping information from scanned ULP "Pink"
shipping documents.

You MUST inspect every page in the attached chunk carefully.

The most important requirement is completeness.

Do not skip a page merely because it resembles another page.

Return every Sales Order visible in the chunk and every genuine
handwritten handling-unit / pallet notation.

PAGE NUMBER RULE

The attached PDF chunk contains at most 3 pages.

Use LOCAL page numbers only:

- first page of this chunk = page 1
- second page of this chunk = page 2
- third page of this chunk = page 3

Do not attempt to determine the page number from the original packet.

Python will convert local page numbers to the original PDF page numbers.


SALES ORDERS

For each Sales Order extract when visible:

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

Sales Order values commonly begin with SO-.

If the Sales Order number is NOT visible on a page:
- DO NOT guess it.
- sales_order must be null.
- Still return the other shipment information if it is visible.

If the same Sales Order is visibly shown on multiple pages in this
chunk, return one object and include all relevant LOCAL page numbers.


HANDLING UNITS

This is the most important extraction task.

Find EVERY genuine handwritten shipping handling-unit or pallet
notation.

There may be:

- one handling unit
- several handling units on one page
- numbered lines such as 1, 2, 3
- faint handwriting
- cramped handwriting
- handwriting overlapping printed content

For every genuine handling unit extract:

- length
- width
- height
- weight
- location
- page

Typical notation may resemble:

48 x 45 x 26 / 97# / B8

or:

1 - 74 x 45 x 55 - 542# - D24


DO NOT CREATE FAKE HANDLING UNITS

Do NOT create a handling-unit object simply because handwriting exists.

A lone location-looking code, packed quantity mark, initials,
signature, note, or miscellaneous handwritten number is NOT enough
to create a handling unit.

Strong evidence of a handling unit normally includes one or more of:

- multiple dimensions
- dimensions plus weight
- dimensions plus location
- a clearly structured pallet notation

If the handwriting is not clearly a handling unit, leave it out.


DIMENSION RULES

Dimensions are normally L x W x H.

- first value = length
- second value = width
- third value = height

Common shipment lengths include:

48
72
74
79
96
98
120
144

If handwriting clearly contains two 48 dimensions and another value,
the intended dimensions may be 48 x 48 x other.

However, use visible evidence first.

Do not force handwriting to match common dimensions.


PRINTED PRODUCT TABLES

Do NOT use printed product dimensions, catalog measurements,
item descriptions, or line-item measurements as pallet dimensions.

Handling-unit dimensions should come from handwritten shipping
notations.


WEIGHT

The # symbol usually indicates pounds.

Example:

97#

means 97 pounds.


LOCATION

Location is normally a short handwritten warehouse or staging code.

Examples:

B8
D24
D22E
07
09
A19-E

Do not append unrelated handwritten numbers or letters.


UNCERTAINTY

Never invent a value.

If a value is unreadable, return null.

Set uncertain=true if any part of a genuine handling unit is
ambiguous.

Use notes to briefly explain the ambiguity.

Do not create numeric confidence percentages.


ADDRESS RULES

Preserve Address 2 separately when clearly present.

A person name, ATTN line, or project name may belong in
delivery_address2.

Do not invent Address 2.


FINAL REVIEW

Before responding:

1. Re-check every page in the chunk.
2. Count visible shipment/order sections.
3. Count genuine handwritten handling-unit notations.
4. Ensure every genuine handling-unit notation is represented.
5. Ensure printed product dimensions were not used.
6. Do not guess Sales Order numbers that are not visible.
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


def _make_pdf_chunk(
    pdf_bytes: bytes,
    start_page: int,
    end_page: int,
) -> bytes:
    """
    start_page:
        zero-based inclusive

    end_page:
        zero-based exclusive
    """

    reader = PdfReader(
        BytesIO(pdf_bytes)
    )

    writer = PdfWriter()

    for page_index in range(
        start_page,
        end_page,
    ):
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
# PAGE NUMBER CONVERSION
# ==========================================================

def _convert_local_page_to_original(
    local_page: Optional[int],
    original_start_page: int,
    original_end_page: int,
) -> Optional[int]:
    """
    GPT returns local chunk pages 1, 2, 3.

    Convert to original PDF page.

    Example:

    chunk = original pages 7-9
    GPT local page 3

    => original page 9
    """

    if local_page is None:
        return None

    try:
        local_page = int(
            local_page
        )
    except Exception:
        return None

    chunk_length = (
        original_end_page
        - original_start_page
        + 1
    )

    if (
        local_page < 1
        or local_page > chunk_length
    ):
        return None

    return (
        original_start_page
        + local_page
        - 1
    )


def _convert_order_pages_to_original(
    order: Dict[str, Any],
    original_start_page: int,
    original_end_page: int,
):
    """
    Convert Sales Order pages and HU pages
    from local chunk numbering to original
    PDF numbering.
    """

    original_pages = []

    for local_page in (
        order.get(
            "pages"
        )
        or []
    ):

        original_page = (
            _convert_local_page_to_original(
                local_page,
                original_start_page,
                original_end_page,
            )
        )

        if (
            original_page is not None
            and original_page not in original_pages
        ):
            original_pages.append(
                original_page
            )

    order[
        "pages"
    ] = original_pages

    for hu in (
        order.get(
            "handling_units"
        )
        or []
    ):

        hu[
            "page"
        ] = (
            _convert_local_page_to_original(
                hu.get(
                    "page"
                ),
                original_start_page,
                original_end_page,
            )
        )


# ==========================================================
# HANDLING UNIT VALIDATION
# ==========================================================

def _count_dimensions(
    hu: Dict[str, Any]
) -> int:

    count = 0

    for field in [
        "length",
        "width",
        "height",
    ]:

        value = hu.get(
            field
        )

        if value is not None:
            count += 1

    return count


def _handling_unit_has_real_data(
    hu: Dict[str, Any]
) -> bool:
    """
    Stronger fake-HU filter.

    Keep if:

    - all 3 dimensions exist

    OR

    - at least 2 dimensions exist
      AND either weight or location exists

    OR

    - at least 1 dimension exists
      AND weight exists
      AND location exists

    This removes weak records such as:

      location = "DC"
      everything else null

    while retaining something like:

      98 x 45 x ? / 294 lb / C21-2
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
# SALES ORDER MERGE HELPERS
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


def _merge_order_into_target(
    target: Dict[str, Any],
    source: Dict[str, Any],
):
    """
    Merge a source record into a target Sales Order.
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

    existing_hu_keys = {
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

        if not _handling_unit_has_real_data(
            hu
        ):
            continue

        key = _handling_unit_key(
            hu
        )

        if key in existing_hu_keys:
            continue

        target[
            "handling_units"
        ].append(
            hu
        )

        existing_hu_keys.add(
            key
        )


# ==========================================================
# NULL-SO MATCHING
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


def _page_distance(
    order_a: Dict[str, Any],
    order_b: Dict[str, Any],
) -> int:
    """
    Minimum distance between any pages.
    """

    pages_a = (
        order_a.get(
            "pages"
        )
        or []
    )

    pages_b = (
        order_b.get(
            "pages"
        )
        or []
    )

    if (
        not pages_a
        or not pages_b
    ):
        return 999999

    return min(
        abs(
            a - b
        )
        for a in pages_a
        for b in pages_b
    )


def _candidate_match_score(
    unidentified: Dict[str, Any],
    identified: Dict[str, Any],
) -> Tuple[int, int]:
    """
    Score null-SO record against known SO.

    Higher score is better.

    Returns:

        (score, page_distance)

    Strong matching signals:

    customer PO = +10
    delivery ZIP = +6
    delivery name = +5
    address = +5
    city = +2
    state = +1

    Nearby pages add confidence indirectly
    through tie breaking.
    """

    score = 0

    if _field_matches(
        unidentified.get(
            "customer_PO"
        ),
        identified.get(
            "customer_PO"
        ),
    ):
        score += 10

    if _field_matches(
        unidentified.get(
            "delivery_zip"
        ),
        identified.get(
            "delivery_zip"
        ),
    ):
        score += 6

    if _field_matches(
        unidentified.get(
            "delivery_name"
        ),
        identified.get(
            "delivery_name"
        ),
    ):
        score += 5

    if _field_matches(
        unidentified.get(
            "delivery_address"
        ),
        identified.get(
            "delivery_address"
        ),
    ):
        score += 5

    if _field_matches(
        unidentified.get(
            "delivery_city"
        ),
        identified.get(
            "delivery_city"
        ),
    ):
        score += 2

    if _field_matches(
        unidentified.get(
            "delivery_state"
        ),
        identified.get(
            "delivery_state"
        ),
    ):
        score += 1

    distance = (
        _page_distance(
            unidentified,
            identified,
        )
    )

    return (
        score,
        distance,
    )


def _attach_unidentified_orders(
    identified_orders: List[Dict[str, Any]],
    unidentified_orders: List[Dict[str, Any]],
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    """
    Attempt to attach null-SO pages to an existing
    identified SO using shipment context.

    We intentionally require a reasonably strong score.

    Accept if:

      score >= 10

    OR

      score >= 8 and page distance <= 2

    Otherwise keep it unidentified.
    """

    unresolved = []

    for unidentified in unidentified_orders:

        best_target = None
        best_score = -1
        best_distance = 999999

        for identified in identified_orders:

            score, distance = (
                _candidate_match_score(
                    unidentified,
                    identified,
                )
            )

            if (
                score > best_score
                or (
                    score == best_score
                    and distance < best_distance
                )
            ):

                best_target = identified
                best_score = score
                best_distance = distance

        should_attach = (
            best_target is not None
            and (
                best_score >= 10
                or (
                    best_score >= 8
                    and best_distance <= 2
                )
            )
        )

        if should_attach:

            _merge_order_into_target(
                best_target,
                unidentified,
            )

        else:

            unresolved.append(
                unidentified
            )

    return (
        identified_orders,
        unresolved,
    )


# ==========================================================
# SALES ORDER MERGE
# ==========================================================

def _merge_sales_orders(
    extracted_orders: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:

    merged_by_so = {}

    unidentified = []

    # ----------------------------------
    # FIRST:
    # Merge explicitly identified SOs
    # ----------------------------------

    for order in extracted_orders:

        sales_order = (
            _clean_string(
                order.get(
                    "sales_order"
                )
            )
        )

        # Clean fake HUs first.
        order[
            "handling_units"
        ] = [
            hu
            for hu in (
                order.get(
                    "handling_units"
                )
                or []
            )
            if _handling_unit_has_real_data(
                hu
            )
        ]

        if not sales_order:

            unidentified.append(
                order
            )

            continue

        key = (
            sales_order.upper()
        )

        if key not in merged_by_so:

            merged_by_so[
                key
            ] = {
                "sales_order":
                    sales_order,

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

        _merge_order_into_target(
            merged_by_so[
                key
            ],
            order,
        )

    identified_orders = list(
        merged_by_so.values()
    )

    # ----------------------------------
    # SECOND:
    # Attach null-SO pages using PO,
    # address, ZIP, consignee, proximity.
    # ----------------------------------

    identified_orders, unresolved = (
        _attach_unidentified_orders(
            identified_orders,
            unidentified,
        )
    )

    # ----------------------------------
    # FINAL CLEANUP / SORTING
    # ----------------------------------

    for order in identified_orders:

        order[
            "pages"
        ] = sorted(
            set(
                order[
                    "pages"
                ]
            )
        )

        order[
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

    identified_orders.sort(
        key=lambda order: (
            min(
                order[
                    "pages"
                ]
            )
            if order[
                "pages"
            ]
            else 999999
        )
    )

    # Keep unresolved records visible for review.
    for order in unresolved:

        order[
            "pages"
        ] = sorted(
            set(
                order.get(
                    "pages"
                )
                or []
            )
        )

        order[
            "handling_units"
        ] = [
            hu
            for hu in (
                order.get(
                    "handling_units"
                )
                or []
            )
            if _handling_unit_has_real_data(
                hu
            )
        ]

    unresolved.sort(
        key=lambda order: (
            min(
                order[
                    "pages"
                ]
            )
            if order[
                "pages"
            ]
            else 999999
        )
    )

    return (
        identified_orders
        + unresolved
    )


# ==========================================================
# ONE GPT CHUNK
# ==========================================================

def _extract_chunk_with_gpt(
    client,
    chunk_pdf_bytes: bytes,
    original_start_page: int,
    original_end_page: int,
):
    """
    GPT returns local page numbers.

    Python converts them afterward.
    """

    uploaded_file_id = None

    try:

        pdf_file = BytesIO(
            chunk_pdf_bytes
        )

        pdf_file.name = (
            f"ulp_chunk_"
            f"{original_start_page}_"
            f"{original_end_page}.pdf"
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
                                    BASE_EXTRACTION_INSTRUCTIONS,
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
                            "ulp_pink_extraction",

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

            extracted = (
                json.loads(
                    output_text
                )
            )

        except json.JSONDecodeError as exc:

            raise RuntimeError(
                "OpenAI returned invalid JSON: "
                f"{str(exc)}"
            )

        # ----------------------------------
        # Convert local pages to original
        # PDF pages deterministically.
        # ----------------------------------

        for order in (
            extracted.get(
                "sales_orders"
            )
            or []
        ):

            _convert_order_pages_to_original(
                order,
                original_start_page,
                original_end_page,
            )

        # ----------------------------------
        # Usage
        # ----------------------------------

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
            extracted,
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
# MAIN EXTRACTOR
# ==========================================================

def extract_ulp_with_gpt(
    pdf_bytes: bytes
):
    """
    GPT ULP extraction pipeline.

    PDF
      -> 3-page chunks
      -> GPT visual extraction
      -> local page numbers
      -> deterministic original page conversion
      -> reject weak fake HUs
      -> merge duplicate Sales Orders
      -> attach null-SO pages via PO/address context
      -> return token usage
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

    all_sales_orders = []

    chunk_results = []

    total_usage = {
        "input_tokens":
            0,

        "output_tokens":
            0,

        "total_tokens":
            0,
    }

    chunk_number = 0

    # ----------------------------------
    # PROCESS EACH 3-PAGE CHUNK
    # ----------------------------------

    for start_index in range(
        0,
        total_pages,
        CHUNK_SIZE,
    ):

        end_index = min(
            start_index + CHUNK_SIZE,
            total_pages,
        )

        chunk_number += 1

        original_start_page = (
            start_index + 1
        )

        original_end_page = (
            end_index
        )

        chunk_pdf = (
            _make_pdf_chunk(
                pdf_bytes,
                start_index,
                end_index,
            )
        )

        extracted, usage = (
            _extract_chunk_with_gpt(
                client=client,
                chunk_pdf_bytes=chunk_pdf,
                original_start_page=
                    original_start_page,
                original_end_page=
                    original_end_page,
            )
        )

        chunk_orders = (
            extracted.get(
                "sales_orders"
            )
            or []
        )

        # ----------------------------------
        # Reject weak HUs immediately.
        # ----------------------------------

        for order in chunk_orders:

            order[
                "handling_units"
            ] = [
                hu
                for hu in (
                    order.get(
                        "handling_units"
                    )
                    or []
                )
                if _handling_unit_has_real_data(
                    hu
                )
            ]

        all_sales_orders.extend(
            chunk_orders
        )

        # ----------------------------------
        # TOTAL USAGE
        # ----------------------------------

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

        # ----------------------------------
        # DEBUG INFO
        # ----------------------------------

        chunk_results.append({
            "chunk":
                chunk_number,

            "pages": [
                original_start_page,
                original_end_page,
            ],

            "sales_order_count":
                len(
                    chunk_orders
                ),

            "usage":
                usage,
        })

    # ----------------------------------
    # MERGE / ASSOCIATE
    # ----------------------------------

    merged_sales_orders = (
        _merge_sales_orders(
            all_sales_orders
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
            CHUNK_SIZE,

        "chunk_count":
            chunk_number,

        "extraction": {
            "sales_orders":
                merged_sales_orders
        },

        "usage":
            total_usage,

        "chunks":
            chunk_results,
    }
