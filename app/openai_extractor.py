import os
import re
import json
from io import BytesIO
from typing import Any, Dict, List, Optional

import fitz
import zxingcpp

from PIL import Image, ImageOps
from openai import OpenAI
from pypdf import PdfReader, PdfWriter


MODEL = "gpt-5.4-mini"
CHUNK_SIZE = 1


# ==========================================================
# BUSINESS RULES
# ==========================================================

# MASTER ULP SALES ORDER.
#
# Examples:
#   SO-00325428
#   SO-00325352
#
# This is NOT the SRP number.
SALES_ORDER_PATTERN = re.compile(
    r"^SO-\d{8}$",
    re.IGNORECASE,
)

SALES_ORDER_SEARCH_PATTERN = re.compile(
    r"SO-\d{8}",
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

If a field is not visible on THIS PAGE, return null.


============================================================
MASTER SALES ORDER — CRITICAL
============================================================

There are multiple identifiers on these documents.

They MUST NOT be confused.

The MASTER ULP Sales Order has this exact format:

SO-########

Example:

SO-00325428
SO-00325352

That means:

SO-
followed by exactly 8 digits.

The master Sales Order may appear beside a printed field labeled:

Sales order

It may also correspond to the barcode on the top Pink sheet.

Do NOT create a master Sales Order from any other identifier.


============================================================
SRP NUMBER — SEPARATE IDENTIFIER
============================================================

SRP_number is a completely separate field.

An SRP number may look similar to a Sales Order.

For example:

SO0316672
S00316672

These are NOT the master ULP Sales Order.

Example:

Master Sales Order:
SO-00325428

SRP Number:
SO0316672

These must remain separate.

NEVER:

- insert a hyphen into an SRP number
- add digits to an SRP number
- remove digits from an SRP number
- transform an SRP number into a master Sales Order
- use SRP_number as a fallback for sales_order

If the master Sales Order is not clearly visible:

sales_order = null

even if an SRP number is visible.


============================================================
FALSE SALES ORDER EXAMPLES
============================================================

Do NOT treat these as master Sales Orders:

SO0316672
S00316672
SO 322733
ORG SO 322733
322733

A master Sales Order must actually appear as:

SO-########


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
SHIP-TO BLOCK
============================================================

Read the complete Ship To block before assigning its lines.

The block may contain:

- organization/company
- project/site name
- person's name
- street address
- city/state/ZIP


============================================================
STREET ADDRESS IDENTIFICATION — IMPORTANT
============================================================

delivery_address should be the line that most clearly represents
the physical street address.

A US street address usually contains:

1. a building/street number
2. a street or route name
3. often a street suffix or route designation

Examples:

16777 FILLMORE ST
3580 SOUTH 144TH STREET
401 N FAIRVIEW AVE
499 W ELM STREET
1135 NM 554

Common street words include:

ST
STREET
RD
ROAD
AVE
AVENUE
BLVD
BOULEVARD
DR
DRIVE
LN
LANE
CT
COURT
HWY
HIGHWAY
PKWY
PARKWAY
WAY
TRL
TRAIL
CIR
CIRCLE
PL
PLACE
TER
TERRACE

State-route style addresses may also look like:

1135 NM 554

A line beginning with a street/building number is a strong clue
that it is delivery_address.


============================================================
ADDRESS 2
============================================================

Lines in the Ship To block that are NOT the company name,
street address, or city/state/ZIP may belong in delivery_address2.

Examples include:

- project name
- site name
- attention line
- recipient/person name
- building or facility name
- suite information

Example Ship To block:

HOA PLAYGROUND SERVICES LLC
QUEENSLAND MANOR
MEAGAN BIRDSALL
3580 SOUTH 144TH STREET
GILBERT, AZ 85297

Correct extraction:

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

Do NOT put a person's name into delivery_contact merely because
the person's name appears in the Ship To address block.


============================================================
DELIVERY CONTACT
============================================================

delivery_contact should normally come from the field explicitly
labeled:

Delivery contact

or the corresponding delivery-contact phone field.

If the Pink says:

Contact phone     630-897-8489
Delivery contact  616-566-7124

then:

delivery_contact = 616-566-7124

Do NOT substitute the general Contact phone when a distinct
Delivery contact value exists.


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
DIMENSIONS
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

Typical widths commonly include:

40
42
44
45
48

These are contextual clues only.

The visible handwriting remains the primary source.


============================================================
NUMBER ROLE CHECK
============================================================

Do NOT assume the first three numbers are automatically dimensions.

Determine whether a number represents:

- length
- width
- height
- weight
- location
- unrelated notation

Example:

48 x 44 / 756# / C27

should mean approximately:

length = 48
width = 44
height = null
weight = 756
location = C27

It should NOT become:

48 x 44 x 756

Also do not split one visible number incorrectly.

Example:

756

should not become:

75 and 6

unless the handwriting clearly supports two separate values.


============================================================
AMBIGUOUS HANDWRITING
============================================================

Be careful with handwritten values such as:

19 vs 91
17 vs 71
14 vs 41
78 vs 98

Use the actual visible pen strokes.

If genuinely ambiguous:

- return the best visual interpretation
- set uncertain=true
- explain the ambiguity briefly in notes

Do not force a preferred dimension merely because it is common.


============================================================
WEIGHT
============================================================

The # symbol commonly means pounds.

Examples:

88# = 88 pounds
294# = 294 pounds
756# = 756 pounds


============================================================
LOCATION
============================================================

Location normally resembles a short warehouse/staging code.

Examples:

B8
C27
D24
D22E
D19E
C21-2
A19-E


============================================================
PRINTED PRODUCT DIMENSIONS
============================================================

Do NOT use printed:

- product measurements
- catalog dimensions
- item descriptions
- line-item measurements

as handling-unit dimensions.


============================================================
DO NOT CREATE FAKE HANDLING UNITS
============================================================

Do not create an HU from:

- initials
- signatures
- check marks
- lone locations
- isolated numbers
- miscellaneous handwriting
- packed quantities

A genuine HU should contain meaningful pallet/shipping structure.


============================================================
INCOMPLETE HANDLING UNIT
============================================================

A real handling unit may have one unreadable field.

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
FINAL REVIEW
============================================================

Before responding:

1. Search for an actual master SO-########.
2. Make sure an SRP number was NOT converted into the Sales Order.
3. Read the entire Ship To block before assigning address fields.
4. Identify the actual physical street-address line.
5. Preserve project/person lines in Address 2 when appropriate.
6. Re-check every handwritten handling unit.
7. Ensure weights were not treated as dimensions.
8. Ensure printed product dimensions were ignored.
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


def _sales_order_from_barcode_text(
    value
) -> Optional[str]:
    """
    Only accept an ACTUAL SO-######## contained in the
    machine-decoded barcode value.

    Do not convert:
        S00316672
        SO0316672
        00325428

    into Sales Orders.
    """

    value = _clean_string(
        value
    )

    if not value:
        return None

    value = value.upper()

    match = SALES_ORDER_SEARCH_PATTERN.search(
        value
    )

    if not match:
        return None

    return _validate_sales_order(
        match.group(0)
    )


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
# BARCODE IMAGE RENDERING
# ==========================================================

def _render_pdf_page(
    page_pdf_bytes: bytes,
    scale: float,
) -> Image.Image:

    document = fitz.open(
        stream=page_pdf_bytes,
        filetype="pdf",
    )

    try:

        page = document[0]

        matrix = fitz.Matrix(
            scale,
            scale,
        )

        pix = page.get_pixmap(
            matrix=matrix,
            alpha=False,
        )

        image = Image.frombytes(
            "RGB",
            [
                pix.width,
                pix.height,
            ],
            pix.samples,
        )

        return image

    finally:

        document.close()


def _top_crop(
    image: Image.Image,
    fraction: float = 0.50,
) -> Image.Image:

    height = int(
        image.height
        * fraction
    )

    return image.crop(
        (
            0,
            0,
            image.width,
            height,
        )
    )


# ==========================================================
# BARCODE DECODING
# ==========================================================

def _decode_image_barcodes(
    image: Image.Image
):
    """
    Decode one raster variant.
    """

    try:

        return zxingcpp.read_barcodes(
            image
        )

    except Exception:

        return []


def _decode_page_barcodes(
    page_pdf_bytes: bytes,
) -> Dict[str, Any]:
    """
    Lightweight barcode pass.

    Try:

    1. full page at 3x
    2. top half at 4x
    3. grayscale top half
    4. autocontrast top half

    This is intentionally lighter than the previous version.
    """

    try:

        image_3x = _render_pdf_page(
            page_pdf_bytes,
            3.0,
        )

        image_4x = _render_pdf_page(
            page_pdf_bytes,
            4.0,
        )

        top_4x = _top_crop(
            image_4x,
            0.50,
        )

        gray_top = ImageOps.grayscale(
            top_4x
        )

        contrast_top = ImageOps.autocontrast(
            gray_top
        )

        variants = [
            (
                "full_3x",
                image_3x,
            ),
            (
                "top_4x",
                top_4x,
            ),
            (
                "gray_top_4x",
                gray_top,
            ),
            (
                "contrast_top_4x",
                contrast_top,
            ),
        ]

        decoded_values = []
        decoded_formats = []
        successful_variants = []

        barcode_so = None

        seen_values = set()

        for variant_name, image in variants:

            results = _decode_image_barcodes(
                image
            )

            if results:

                successful_variants.append(
                    variant_name
                )

            for result in results:

                text = _clean_string(
                    getattr(
                        result,
                        "text",
                        None,
                    )
                )

                if not text:
                    continue

                if text not in seen_values:

                    seen_values.add(
                        text
                    )

                    decoded_values.append(
                        text
                    )

                fmt = getattr(
                    result,
                    "format",
                    None,
                )

                if fmt is not None:

                    fmt_text = str(
                        fmt
                    )

                    if fmt_text not in decoded_formats:

                        decoded_formats.append(
                            fmt_text
                        )

                if barcode_so is None:

                    candidate = (
                        _sales_order_from_barcode_text(
                            text
                        )
                    )

                    if candidate:

                        barcode_so = candidate

        return {
            "sales_order":
                barcode_so,

            "values":
                decoded_values,

            "formats":
                decoded_formats,

            "successful_variants":
                successful_variants,

            "error":
                None,
        }

    except Exception as exc:

        return {
            "sales_order":
                None,

            "values":
                [],

            "formats":
                [],

            "successful_variants":
                [],

            "error":
                str(
                    exc
                ),
        }


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
    # Example:
    #
    # 48 x 44 x 756
    #
    # 756 is almost certainly weight, not height.
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
    # Possible 756 -> 75 + 6 split.
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
    # Model says:
    #
    # height=75
    # weight=756
    #
    # but notes say height is unclear.
    #
    # Remove the questionable height.
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

        if (
            "height is not clearly" in note_text
            or "height is unclear" in note_text
            or "height is not legible" in note_text
            or "height is unreadable" in note_text
            or (
                "height" in note_text
                and "unclear" in note_text
            )
        ):

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
                    f"because the model identified the height "
                    f"as unclear while weight {weight} was "
                    f"independently identified."
                )
            )

    # ------------------------------------------------------
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

    # Different known SOs:
    # hard shipment boundary.
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

    # Valid SO appears after unidentified shipment pages.
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

    barcode_stats = {
        "pages_with_barcodes": 0,
        "sales_orders_found": 0,
        "decode_errors": 0,
    }

    # ======================================================
    # ONE PAGE PER GPT CALL
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

        # --------------------------------------------------
        # BARCODE
        # --------------------------------------------------

        barcode_result = _decode_page_barcodes(
            page_pdf
        )

        barcode_so = _validate_sales_order(
            barcode_result.get(
                "sales_order"
            )
        )

        barcode_values = (
            barcode_result.get(
                "values"
            )
            or []
        )

        if barcode_values:

            barcode_stats[
                "pages_with_barcodes"
            ] += 1

        if barcode_so:

            barcode_stats[
                "sales_orders_found"
            ] += 1

        if barcode_result.get(
            "error"
        ):

            barcode_stats[
                "decode_errors"
            ] += 1

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
            )
        )

        gpt_so = _validate_sales_order(
            page_result.get(
                "sales_order"
            )
        )

        # --------------------------------------------------
        # MASTER SALES ORDER PRIORITY
        #
        # BARCODE WINS IF IT DECODED AN ACTUAL SO-########.
        #
        # OTHERWISE USE STRICTLY VALID GPT READING.
        #
        # SRP IS NEVER USED AS FALLBACK.
        # --------------------------------------------------

        if barcode_so:

            final_so = barcode_so

        else:

            final_so = gpt_so

        page_result[
            "sales_order"
        ] = final_so

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
                final_so,

            "gpt_sales_order":
                gpt_so,

            "barcode_sales_order":
                barcode_so,

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

            "barcode_values":
                barcode_values,

            "barcode_formats":
                barcode_result.get(
                    "formats"
                )
                or [],

            "barcode_successful_variants":
                barcode_result.get(
                    "successful_variants"
                )
                or [],

            "barcode_error":
                barcode_result.get(
                    "error"
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
    # GROUP INTO SHIPMENTS
    # ======================================================

    grouped_shipments = _group_pages_into_shipments(
        page_results
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

        "barcode": {

            "pages_with_barcodes":
                barcode_stats[
                    "pages_with_barcodes"
                ],

            "sales_orders_found":
                barcode_stats[
                    "sales_orders_found"
                ],

            "decode_errors":
                barcode_stats[
                    "decode_errors"
                ],
        },

        "pages":
            page_debug,
    }
