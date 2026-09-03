import os
import re
import json
from io import BytesIO
from typing import Any, Dict, List, Optional

from openai import OpenAI
from pypdf import PdfReader, PdfWriter


MODEL = "gpt-5.4-mini"
CHUNK_SIZE = 1


# ==========================================================
# BUSINESS RULES
# ==========================================================

SALES_ORDER_PATTERN = re.compile(
    r"^SO-\d{8}$",
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
# MAIN PAGE RESPONSE SCHEMA
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
# SALES ORDER RECOVERY SCHEMA
# ==========================================================

SALES_ORDER_ONLY_SCHEMA = {
    "type": "object",
    "properties": {
        "sales_order": {
            "type": ["string", "null"]
        }
    },
    "required": [
        "sales_order"
    ],
    "additionalProperties": False
}


# ==========================================================
# MAIN PAGE PROMPT
# ==========================================================

EXTRACTION_INSTRUCTIONS = """
You are extracting shipping information from exactly ONE scanned
ULP "Pink" shipping document page.

Inspect the entire visual page carefully.

Do NOT infer information from previous or following pages.

If something is not visible on THIS PAGE, return null.


============================================================
SALES ORDER
============================================================

Actively search the entire page for a valid ULP Sales Order.

Valid ULP Sales Orders look exactly like:

SO-00325355

Format:

SO-
followed by exactly 8 digits.

The Sales Order may appear:

- next to the printed label "Sales order"
- in the upper portion of a Pink sheet
- in a barcode/header region
- on a packing list

Do NOT return unrelated references such as:

SO 322733
ORG SO 322733
322733

If you cannot confidently find a valid SO-######## on this page,
return null.

Never infer Sales Order from customer PO or consignee.


============================================================
SHIPMENT INFORMATION
============================================================

When visibly present, extract:

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


============================================================
ADDRESS RULES
============================================================

delivery_name:
business, institution, school, municipality, customer, or destination.

delivery_address:
primary street address.

delivery_address2:
ATTN/person name, project name, suite, secondary address information.

Do not invent Address 2.


============================================================
HANDLING UNITS
============================================================

Find EVERY genuine handwritten pallet / handling-unit notation.

Examples:

48 x 45 x 26 / 97# / B8

74 x 45 x 55 / 542# / D24

48 x 48 x 17 / 88# / D19E

Extract:

- length
- width
- height
- weight
- location


============================================================
DIMENSIONS / WEIGHT
============================================================

Dimensions normally appear:

L x W x H

Typical lengths:

48
72
74
79
96
98
120
144

Typical widths:

40
42
44
45
48

The # symbol commonly means pounds.

Do not automatically treat the first three numbers as dimensions.

For example:

48 x 44 / 756# / C27

should not become:

48 x 44 x 756

Also do not split:

756

into:

75 and 6

unless the handwriting clearly supports that.


============================================================
AMBIGUITY
============================================================

Be careful with values such as:

19 vs 91
17 vs 71
14 vs 41
24 vs 74

If genuinely ambiguous:

- use the most likely visible interpretation
- set uncertain=true
- explain briefly in notes


============================================================
PRINTED PRODUCT TABLES
============================================================

Do NOT use printed catalog/product dimensions as pallet dimensions.


============================================================
DO NOT CREATE FAKE HANDLING UNITS
============================================================

Do not create an HU from:

- initials
- signatures
- check marks
- lone locations
- one isolated number
- miscellaneous handwriting
- packed quantities

A genuine HU should have meaningful pallet structure.


============================================================
INCOMPLETE HANDLING UNIT
============================================================

A real HU may still have one unreadable field.

Example:

98 x 45 x ? / 294# / C21-2

Return:

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

1. Search again for a valid SO-########.
2. Re-scan all handwriting for pallet notation.
3. Confirm weights were not mistaken for dimensions.
4. Confirm printed product dimensions were ignored.
5. Confirm random handwriting was not turned into a pallet.
"""


# ==========================================================
# TARGETED SALES ORDER PROMPT
# ==========================================================

SALES_ORDER_RECOVERY_INSTRUCTIONS = """
Your ONLY job is to find the ULP Sales Order on this single page.

Inspect the page very carefully.

A valid Sales Order has exactly this format:

SO-########

Example:

SO-00325352

Search specifically:

1. next to the printed label "Sales order"
2. the upper/header portion of the page
3. the barcode area
4. human-readable text associated with the barcode

On a ULP Pink sheet, the barcode corresponds to the Sales Order.

Do NOT return:

- Customer PO
- customer reference
- packing-list number
- ORG SO notes
- values like "SO 322733"

Only return a value if it matches:

SO-
followed by exactly 8 digits.

If no valid Sales Order is actually visible, return null.
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
# STRING HELPERS
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
# SALES ORDER VALIDATION
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
# DETERMINE WHETHER SO RECOVERY IS WORTH RUNNING
# ==========================================================

def _should_attempt_so_recovery(
    page_result: Dict[str, Any]
) -> bool:

    if _validate_sales_order(
        page_result.get(
            "sales_order"
        )
    ):
        return False

    # Only run the extra call when the page contains
    # meaningful shipment/header information.
    return any(
        _clean_string(
            page_result.get(
                field
            )
        )
        for field in [
            "customer_PO",
            "delivery_name",
            "delivery_address",
            "delivery_zip",
            "SRP_number",
        ]
    )


# ==========================================================
# TARGETED SALES ORDER RECOVERY
# ==========================================================

def _recover_sales_order_with_gpt(
    client,
    page_pdf_bytes: bytes,
    original_page_number: int,
):

    uploaded_file_id = None

    try:

        pdf_file = BytesIO(
            page_pdf_bytes
        )

        pdf_file.name = (
            f"ulp_so_recovery_"
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
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text":
                                    SALES_ORDER_RECOVERY_INSTRUCTIONS,
                            },
                            {
                                "type": "input_file",
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
                            "ulp_sales_order_recovery",

                        "strict":
                            True,

                        "schema":
                            SALES_ORDER_ONLY_SCHEMA,
                    }
                },
            )
        )

        output_text = (
            response.output_text
            or ""
        ).strip()

        if not output_text:

            return (
                None,
                {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                },
            )

        try:

            extracted = json.loads(
                output_text
            )

        except json.JSONDecodeError:

            return (
                None,
                {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                },
            )

        sales_order = (
            _validate_sales_order(
                extracted.get(
                    "sales_order"
                )
            )
        )

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
            sales_order,
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

    # 48 x 44 x 756 -> likely 756 lb
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
                f"is much more plausible as weight. "
                f"Height requires review."
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

    # Possible 756 -> 75 + 6 split
    if (
        uncertain
        and height is not None
        and 60 <= height <= 120
        and weight is not None
        and 0 < weight <= 10
        and length is not None
        and width is not None
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
                f"height={bad_height} and weight={bad_weight} "
                f"were not trusted. Verify original notation."
            )
        )

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
                    f"{field}={value} is outside the "
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
        or (
            40 <= length <= 160
        )
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
# MAIN GPT PAGE EXTRACTION
# ==========================================================

def _extract_page_with_gpt(
    client,
    page_pdf_bytes: bytes,
    original_page_number: int,
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
                        "role": "user",
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
        ] = (
            original_page_number
        )

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
# PAGE SIMILARITY
# ==========================================================

def _page_similarity_score(
    a: Dict[str, Any],
    b: Dict[str, Any],
) -> int:

    score = 0

    if _field_matches(
        a.get("customer_PO"),
        b.get("customer_PO"),
    ):
        score += 12

    if _field_matches(
        a.get("delivery_zip"),
        b.get("delivery_zip"),
    ):
        score += 6

    if _field_matches(
        a.get("delivery_name"),
        b.get("delivery_name"),
    ):
        score += 5

    if _field_matches(
        a.get("delivery_address"),
        b.get("delivery_address"),
    ):
        score += 5

    if _field_matches(
        a.get("delivery_city"),
        b.get("delivery_city"),
    ):
        score += 2

    if _field_matches(
        a.get("delivery_state"),
        b.get("delivery_state"),
    ):
        score += 1

    if _field_matches(
        a.get("delivery_contact"),
        b.get("delivery_contact"),
    ):
        score += 2

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
        "sales_order": None,
        "pages": [],
        "customer_PO": None,
        "SRP_number": None,
        "delivery_name": None,
        "delivery_address": None,
        "delivery_address2": None,
        "delivery_city": None,
        "delivery_state": None,
        "delivery_zip": None,
        "delivery_contact": None,
        "handling_units": [],
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
        and current_so
        != group_so
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
# BACKWARD SO STITCHING
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

    return identity_count <= 1


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

    if len(groups) < 2:
        return groups

    stitched = []

    i = 0

    while i < len(groups):

        current = groups[
            i
        ]

        if (
            i + 1
            < len(groups)
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
# GROUP PAGES
# ==========================================================

def _group_pages_into_shipments(
    page_results: List[
        Dict[str, Any]
    ]
) -> List[
    Dict[str, Any]
]:

    groups = []

    current_group = _new_shipment_group()

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

            current_group = _new_shipment_group()

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
# MAIN EXTRACTOR
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

    client = _get_client()

    page_results = []
    page_debug = []

    total_usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }

    recovery_usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "attempts": 0,
        "successes": 0,
    }

    # ======================================================
    # ONE PAGE AT A TIME
    # ======================================================

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

        page_result, usage = (
            _extract_page_with_gpt(
                client=
                    client,

                page_pdf_bytes=
                    page_pdf,

                original_page_number=
                    page_number,
            )
        )

        # ==================================================
        # TARGETED SO RECOVERY
        # ==================================================

        recovered_so = None

        if _should_attempt_so_recovery(
            page_result
        ):

            recovery_usage[
                "attempts"
            ] += 1

            recovered_so, so_usage = (
                _recover_sales_order_with_gpt(
                    client=
                        client,

                    page_pdf_bytes=
                        page_pdf,

                    original_page_number=
                        page_number,
                )
            )

            recovery_usage[
                "input_tokens"
            ] += so_usage[
                "input_tokens"
            ]

            recovery_usage[
                "output_tokens"
            ] += so_usage[
                "output_tokens"
            ]

            recovery_usage[
                "total_tokens"
            ] += so_usage[
                "total_tokens"
            ]

            if recovered_so:

                page_result[
                    "sales_order"
                ] = recovered_so

                recovery_usage[
                    "successes"
                ] += 1

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
                page_result.get(
                    "sales_order"
                ),

            "sales_order_recovered":
                recovered_so,

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

    # Include recovery tokens in grand total.
    total_usage[
        "input_tokens"
    ] += recovery_usage[
        "input_tokens"
    ]

    total_usage[
        "output_tokens"
    ] += recovery_usage[
        "output_tokens"
    ]

    total_usage[
        "total_tokens"
    ] += recovery_usage[
        "total_tokens"
    ]

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

        "sales_order_recovery": {
            "attempts":
                recovery_usage[
                    "attempts"
                ],

            "successes":
                recovery_usage[
                    "successes"
                ],

            "input_tokens":
                recovery_usage[
                    "input_tokens"
                ],

            "output_tokens":
                recovery_usage[
                    "output_tokens"
                ],

            "total_tokens":
                recovery_usage[
                    "total_tokens"
                ],
        },

        "pages":
            page_debug,
    }
