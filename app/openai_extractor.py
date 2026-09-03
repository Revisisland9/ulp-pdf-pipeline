import os
import json
from io import BytesIO
from typing import Any, Dict, List

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

You MUST carefully inspect every page provided.

The most important requirement is completeness.

Do not skip a page simply because it appears similar to another page.

Return every Sales Order visible in this chunk and every genuine
handwritten handling-unit / pallet notation.

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

A Sales Order may continue across multiple pages.

If the same Sales Order appears on multiple pages in this chunk,
return ONE Sales Order object and include all relevant page numbers.

Customer reference / requisition / PO information should be placed
in customer_PO when it represents the customer's purchase order.


HANDLING UNITS

This is the most important part.

Find EVERY genuine handwritten shipping handling-unit or pallet
notation.

There may be:

- one handling unit
- several handling units on one page
- numbered handwritten lines such as 1, 2, 3, etc.
- faint or cramped handwritten dimensions
- handwriting overlapping printed text

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


IMPORTANT HANDLING-UNIT RULE

Do NOT create a handling-unit object merely because handwriting
exists on the page.

Only return a handling unit when there is real evidence that the
handwriting represents a pallet / handling unit / shipping unit.

If handwriting exists but no genuine handling unit can be identified,
return an empty handling_units array for that Sales Order.


DIMENSION RULES

- Dimensions are normally L x W x H.
- First number is normally length.
- Second number is normally width.
- Third number is normally height.

Common shipment lengths include:

48
72
74
79
96
98
120
144

If handwriting clearly contains two 48 dimensions plus another value,
the intended dimensions may be 48 x 48 x other.

However, use the visible evidence first.

Do not force dimensions to match common values if the handwriting
clearly says something different.


PRINTED ITEM TABLES

VERY IMPORTANT:

Do NOT interpret printed product dimensions, item dimensions,
catalog dimensions, or printed line-item measurements as pallet
dimensions.

Handling-unit dimensions should come from handwritten shipping
notations.


WEIGHT

The # symbol commonly indicates pounds.

Example:

97#

means approximately 97 pounds.


LOCATION

Location is normally a short handwritten warehouse or staging code.

Examples may resemble:

B8
D24
D22E
07
09
A19-E

Do not append unrelated handwritten numbers to the location.


UNCERTAINTY

Never invent a value.

If a field cannot be read, return null.

Set uncertain=true if any part of a handling unit is genuinely
ambiguous.

Use notes to explain the ambiguity briefly.

Do NOT manufacture numeric confidence percentages.


ADDRESS RULES

Preserve Address 2 separately when it is clearly present.

An ATTN/person/project line may belong in delivery_address2 rather
than delivery_name.

Do not invent an Address 2.


FINAL REVIEW

Before responding:

1. Re-check EVERY page in this chunk.
2. Count the visible Sales Orders.
3. Count genuine handwritten handling-unit lines.
4. Verify every genuine handling-unit line is represented.
5. Verify printed item dimensions were not mistaken for pallet dimensions.
6. Verify all handling units are attached to the correct Sales Order.
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
    start_page is zero-based inclusive.
    end_page is zero-based exclusive.
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
# VALUE HELPERS
# ==========================================================

def _clean_string(
    value
):

    if value is None:
        return None

    value = str(
        value
    ).strip()

    return value or None


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
# HANDLING UNIT CLEANUP
# ==========================================================

def _handling_unit_has_real_data(
    hu: Dict[str, Any]
) -> bool:
    """
    Reject pseudo-HUs where the model only said:
    'I saw handwriting but couldn't identify anything.'

    At least one meaningful shipping value must exist.
    """

    meaningful_fields = [
        hu.get("length"),
        hu.get("width"),
        hu.get("height"),
        hu.get("weight"),
        _clean_string(
            hu.get("location")
        ),
    ]

    return any(
        value not in (
            None,
            "",
        )
        for value in meaningful_fields
    )


def _handling_unit_key(
    hu: Dict[str, Any]
):
    """
    Used for deduplicating identical HUs.
    """

    return (
        hu.get("page"),
        hu.get("length"),
        hu.get("width"),
        hu.get("height"),
        hu.get("weight"),
        _clean_string(
            hu.get("location")
        ),
    )


# ==========================================================
# SALES ORDER MERGE
# ==========================================================

def _merge_sales_orders(
    extracted_orders: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Merge duplicate Sales Order objects created by separate
    chunks or multiple pages.

    Primary key:
        sales_order

    If sales_order is missing, the record is retained separately.
    """

    merged = {}

    unidentified = []

    shipment_fields = [
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

    for order in extracted_orders:

        sales_order = _clean_string(
            order.get(
                "sales_order"
            )
        )

        if not sales_order:

            unidentified.append(
                order
            )

            continue

        key = sales_order.upper()

        if key not in merged:

            merged[
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

        target = merged[
            key
        ]

        # ----------------------------------
        # Merge pages
        # ----------------------------------

        for page in (
            order.get(
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

        # ----------------------------------
        # Merge shipment-level fields
        # ----------------------------------

        for field in shipment_fields:

            target[
                field
            ] = _first_nonempty(
                target.get(
                    field
                ),
                order.get(
                    field
                ),
            )

        # ----------------------------------
        # Merge handling units
        # ----------------------------------

        existing_keys = {
            _handling_unit_key(
                hu
            )
            for hu in target[
                "handling_units"
            ]
        }

        for hu in (
            order.get(
                "handling_units"
            )
            or []
        ):

            if not _handling_unit_has_real_data(
                hu
            ):
                continue

            hu_key = _handling_unit_key(
                hu
            )

            if hu_key in existing_keys:
                continue

            target[
                "handling_units"
            ].append(
                hu
            )

            existing_keys.add(
                hu_key
            )

    result = list(
        merged.values()
    )

    # ----------------------------------
    # Sort pages / HUs
    # ----------------------------------

    for order in result:

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

    # ----------------------------------
    # Sort Sales Orders by first page
    # ----------------------------------

    result.sort(
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

    result.extend(
        unidentified
    )

    return result


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
    Send one 3-page chunk to GPT.

    original_start_page and original_end_page
    are one-based page numbers.
    """

    uploaded_file_id = None

    try:

        pdf_file = BytesIO(
            chunk_pdf_bytes
        )

        pdf_file.name = (
            f"ulp_pages_"
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

        page_instruction = f"""
The attached PDF contains ORIGINAL DOCUMENT PAGES
{original_start_page} through {original_end_page}.

IMPORTANT PAGE NUMBER RULE:

When you return a page number, use the ORIGINAL PDF page number.

Do NOT renumber the attached chunk starting at page 1.

For example, if this chunk contains original pages
{original_start_page}-{original_end_page}, any handling-unit page
must use those original page numbers.
"""

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
                                    (
                                        BASE_EXTRACTION_INSTRUCTIONS
                                        + "\n\n"
                                        + page_instruction
                                    ),
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

            extracted = json.loads(
                output_text
            )

        except json.JSONDecodeError as exc:

            raise RuntimeError(
                "OpenAI returned invalid JSON: "
                f"{str(exc)}"
            )

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
      -> split into 3-page chunks
      -> GPT extraction for each chunk
      -> drop empty fake HUs
      -> merge duplicate Sales Orders
      -> deduplicate handling units
      -> total token usage
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
    # PROCESS CHUNKS
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

        # Human-facing original pages.
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
        # Clean pseudo-HUs immediately
        # ----------------------------------

        for order in chunk_orders:

            handling_units = (
                order.get(
                    "handling_units"
                )
                or []
            )

            order[
                "handling_units"
            ] = [
                hu
                for hu in handling_units
                if _handling_unit_has_real_data(
                    hu
                )
            ]

        all_sales_orders.extend(
            chunk_orders
        )

        # ----------------------------------
        # Usage
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
        # Debug information
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
    # MERGE EVERYTHING
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
