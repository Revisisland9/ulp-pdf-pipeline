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
# GPT INSTRUCTIONS
# ==========================================================

EXTRACTION_INSTRUCTIONS = """
You are extracting shipping information from exactly ONE scanned
ULP "Pink" shipping document page.

Inspect the entire visual page carefully.

Do NOT infer information from previous or following pages.

If something is not visible on THIS PAGE, return null.


============================================================
SALES ORDER — IMPORTANT
============================================================

Actively search the ENTIRE PAGE for a Sales Order number.

Valid ULP Sales Orders normally look exactly like:

SO-00325355

In other words:

SO-
followed by exactly 8 digits.

They may appear:

- near the top or upper-right
- on a yellow PACKING LIST
- beside or underneath the words "Sales Order"
- in smaller printed text than the main shipment information

Before returning sales_order=null, make one final visual search of
the entire page specifically for a valid value beginning with "SO-".

Do NOT return values such as:

SO 322733
ORG SO 322733
322733

Those may be notes or references, not the true Sales Order.

Only return a Sales Order when it is actually visible and matches the
ULP Sales Order format.

Never guess a Sales Order from a customer PO, customer reference,
packing-list number, consignee, or other field.


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

Preserve visible wording faithfully.


============================================================
ADDRESS RULES
============================================================

delivery_name:
the business, organization, institution, school, municipality,
customer, or destination name.

delivery_address:
primary street address.

delivery_address2:
ATTN/person name, project name, suite, secondary address information,
or other clearly secondary destination information.

Do not invent Address 2.


============================================================
HANDLING UNITS
============================================================

This is the most important visual extraction task.

Find EVERY genuine handwritten pallet / handling-unit notation.

Typical examples:

48 x 45 x 26 / 97# / B8

74 x 45 x 55 / 542# / D24

48 x 48 x 17 / 88# / D19E

There may be more than one handling unit on the page.

For every genuine handling unit extract:

- length
- width
- height
- weight
- location


============================================================
DIMENSION ORDER
============================================================

Dimensions normally appear:

L x W x H

Typical shipment lengths include:

48
72
74
79
96
98
120
144

Typical pallet widths are commonly around:

40
42
44
45
48

These are contextual clues only.

The visual handwriting is the primary source.


============================================================
IMPORTANT NUMBER-ROLE CHECK
============================================================

When reading a handwritten sequence, determine whether each number
is actually:

- a dimension
- a weight
- a location
- another unrelated number

Do NOT automatically treat the first three visible numbers as
L x W x H.

For example:

48 x 44 / 756# / C27

should NOT become:

48 x 44 x 756

because 756 is much more plausible as shipment weight than as a
756-inch pallet height.

Also be careful not to split one handwritten number incorrectly.

For example:

756

should not casually become:

height = 75
weight = 6

Look carefully for:

- #
- lb / lbs
- dashes
- slashes
- spacing
- visual grouping
- placement next to the location code


============================================================
AMBIGUOUS HANDWRITING
============================================================

Be particularly careful with visually reversible handwritten values,
for example:

19 versus 91
17 versus 71
14 versus 41
24 versus 74

Use the actual pen strokes and surrounding shipping context.

If still ambiguous:

- return the most likely visible interpretation
- set uncertain=true
- explain the ambiguity briefly in notes

Do not silently transpose digits only because another number seems
more common.


============================================================
TWO 48 RULE
============================================================

If handwriting clearly shows two 48 dimensions and another
dimension, the intended orientation may be:

48 x 48 x other

Use this as context, not as permission to override clearly visible
handwriting.


============================================================
WEIGHT
============================================================

The # symbol commonly indicates pounds.

Examples:

88# = 88 pounds
294# = 294 pounds
756# = 756 pounds

Do not confuse quantity or product numbers with pallet weight.


============================================================
LOCATION
============================================================

Location is normally a short handwritten warehouse / staging code.

Examples:

B8
D24
D22E
D19E
C21-2
C27
07
09
A19-E

Do not append unrelated handwriting.


============================================================
PRINTED PRODUCT TABLES
============================================================

Do NOT use printed product dimensions, catalog measurements,
item descriptions, quantities, or product-table measurements as
handling-unit dimensions.

Handling-unit measurements must come from genuine handwritten
shipping notation.


============================================================
DO NOT CREATE FAKE HANDLING UNITS
============================================================

Handwriting alone does NOT mean a handling unit exists.

Do not create a handling-unit object from:

- initials
- signatures
- packed quantities
- check marks
- a lone location code
- one isolated number
- random notes
- marks in Packed / Back Order / Sign columns

A genuine handling unit normally has meaningful structure such as:

- multiple dimensions
- dimensions plus weight
- dimensions plus location
- clearly organized pallet notation


============================================================
INCOMPLETE HANDLING UNITS
============================================================

A real pallet may still have one unreadable value.

Example:

98 x 45 x ? / 294# / C21-2

Return:

length = 98
width = 45
height = null
weight = 294
location = C21-2
uncertain = true

Do not discard an otherwise genuine handling unit merely because
one component is unreadable.


============================================================
UNCERTAINTY
============================================================

Never invent unreadable values.

Use null for unreadable values.

Set uncertain=true whenever a genuine handling-unit value is
ambiguous.

Use notes to briefly explain why.

Do not manufacture numeric confidence percentages.


============================================================
FINAL PAGE CHECK
============================================================

Before responding:

1. Re-scan the entire page specifically for a valid Sales Order
   formatted SO-########.
2. Re-scan the entire page for handwritten pallet notation.
3. Count the genuine handling units.
4. Verify every genuine handling unit was returned.
5. Verify weights were not mistaken for dimensions.
6. Verify product-table dimensions were ignored.
7. Verify miscellaneous handwriting was not converted into a pallet.
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
    """
    Accept only true ULP Sales Orders.

    Accept:
        SO-00325433

    Reject:
        SO 322733
        ORG SO 322733
        322733
    """

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
    """
    Conservative freight plausibility rules.

    GPT reads the handwriting.
    Python prevents clearly bad number-role assignments.
    """

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
    # CASE 1
    #
    # 48 x 44 x 756
    # weight = null
    #
    # Treat 756 as likely weight.
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
                f"is much more plausible as weight. "
                f"Height requires review."
            )
        )

    # Refresh values.
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
    # CASE 2
    #
    # Possible split of one handwritten number:
    #
    # height = 75
    # weight = 6
    #
    # when model itself says uncertain.
    # ------------------------------------------------------

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
                f"were not trusted. Verify the original notation."
            )
        )

    # ------------------------------------------------------
    # CASE 3
    # Extreme dimensions.
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
                    f"{field}={value} is outside the "
                    f"normal ULP handling-unit range."
                )
            )

    # ------------------------------------------------------
    # CASE 4
    # Unusually tall freight.
    # ------------------------------------------------------

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
    """
    Preserve strong or plausibly incomplete handling units.
    Reject random handwriting.
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

    # Full dimensional notation.
    if dimension_count >= 3:
        return True

    # Two dims + weight = strong evidence.
    if (
        dimension_count >= 2
        and has_weight
    ):
        return True

    # Two dims + location:
    # keep only if L/W pair is plausible.
    if (
        dimension_count >= 2
        and has_location
        and _partial_dimensions_are_plausible(
            hu
        )
    ):
        return True

    # One dimension + weight + location
    # still deserves review.
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

        # --------------------------------------------------
        # STRICT SALES ORDER VALIDATION
        # --------------------------------------------------

        page_result[
            "sales_order"
        ] = _validate_sales_order(
            page_result.get(
                "sales_order"
            )
        )

        # Python owns exact page numbering.
        page_result[
            "page"
        ] = (
            original_page_number
        )

        # --------------------------------------------------
        # CLEAN / REPAIR HUs
        # --------------------------------------------------

        cleaned_hus = []

        for hu in (
            page_result.get(
                "handling_units"
            )
            or []
        ):

            hu[
                "page"
            ] = (
                original_page_number
            )

            hu = (
                _repair_handling_unit(
                    hu
                )
            )

            if (
                _handling_unit_has_real_data(
                    hu
                )
            ):

                cleaned_hus.append(
                    hu
                )

        page_result[
            "handling_units"
        ] = cleaned_hus

        # --------------------------------------------------
        # USAGE
        # --------------------------------------------------

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

        key = (
            _handling_unit_key(
                hu
            )
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
# FIRST PASS GROUPING
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

    # ----------------------------------
    # Hard boundary:
    # two different valid SOs.
    # ----------------------------------

    if (
        current_so
        and group_so
        and current_so
        != group_so
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

    # Strong same-shipment identity.
    if similarity >= 8:
        return False

    # Current page introduces a valid Sales Order
    # but prior group has none.
    #
    # Start new temporarily.
    # Backward stitching can attach it safely.
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

    # No recognizable shipment identity:
    # probably a continuation form.
    if not has_identity:
        return False

    # Recognizable but nonmatching identity:
    # likely new shipment.
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

        key = (
            _handling_unit_key(
                hu
            )
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

    # Attach valid SO-only packing list
    # backward to prior unidentified shipment.
    groups = (
        _backward_stitch_sales_orders(
            groups
        )
    )

    # Final sorting.
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

        page_number = (
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
                client=
                    client,

                page_pdf_bytes=
                    page_pdf,

                original_page_number=
                    page_number,
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
                page_number,

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
    # PYTHON ASSEMBLES SHIPMENTS
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

        "pages":
            page_debug,
    }
