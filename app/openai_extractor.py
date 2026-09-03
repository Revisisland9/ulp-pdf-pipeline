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

# MASTER ULP SALES ORDER
#
# Examples:
#   SO-00325428
#   SO-00325352
#
# SRP numbers are separate and must NEVER be transformed
# into this format.
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
ULP shipping document page.

These documents are often called "Pinks."

Inspect the entire visual page carefully.

Do NOT infer information from previous or following pages.

If a field is not visible on THIS PAGE, return null.


============================================================
MASTER SALES ORDER — IMPORTANT
============================================================

On the pink Pro Forma Ship List Report, there is a printed field
in the upper-right information column labeled:

Sales order

Your job is to TRANSCRIBE the value printed beside the literal
field label:

Sales order

Do NOT search the document for a number that merely resembles a
Sales Order.

Locate the actual printed field label:

Sales order

and read its corresponding printed value.


Example:

Sales order     SO-00325352

returns:

sales_order = "SO-00325352"


Another example:

Sales order     SO-00325428

returns:

sales_order = "SO-00325428"


The Sales order field appears in the same general printed
information block as fields such as:

DLV Term
Customer PO
Customer Ref
Site contact
Contact phone
Delivery contact
Ship Date
Ship Via

Treat "Sales order" as a normal labeled printed form field.

If you can read Customer PO, Contact phone, Delivery contact,
or similar nearby printed fields, make a deliberate effort to
also inspect the row labeled "Sales order".


============================================================
MASTER SALES ORDER FORMAT
============================================================

The normal master ULP Sales Order format is:

SO-########

That means:

SO-
followed by exactly 8 digits.

Examples:

SO-00325352
SO-00325428
SO-00325433


After reading the actual printed Sales order field, return the
value only if it matches this format.

Do NOT:

- reconstruct a Sales Order
- normalize another identifier into a Sales Order
- add a hyphen to another identifier
- infer the Sales Order from the customer PO
- infer the Sales Order from the SRP number
- infer the Sales Order from another page


If the actual printed Sales order field cannot be read:

sales_order = null


============================================================
SRP NUMBER — COMPLETELY SEPARATE
============================================================

SRP_number is a different identifier from the master Sales Order.

An SRP number may look similar to a Sales Order.

Examples:

SO0316672
S00316672

These are NOT master ULP Sales Orders.


Example:

Printed Sales order:
SO-00325428

SRP number:
SO0316672

Correct result:

sales_order = "SO-00325428"
SRP_number = "SO0316672"


NEVER transform:

SO0316672

into:

SO-00316672


NEVER transform:

S00316672

into:

SO-00316672


NEVER use SRP_number as a fallback for sales_order.


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
CUSTOMER PO
============================================================

customer_PO should come from the printed field labeled:

Customer PO

or equivalent customer reference when clearly identified as the
customer's purchase order.

Preserve the printed value.


============================================================
SHIP-TO BLOCK
============================================================

Read the COMPLETE Ship To block before assigning its lines.

A Ship To block may contain:

- organization/company name
- project or site name
- recipient/person name
- physical street address
- city/state/ZIP


Do not assume that the second line is always the street address.

First determine what each line represents.


============================================================
STREET ADDRESS IDENTIFICATION — IMPORTANT
============================================================

delivery_address should be the line that most clearly represents
the physical street address.

A US street address normally contains:

1. a building/street number
2. followed by a street, road, route, or highway name
3. often followed by a street suffix


Examples:

16777 FILLMORE ST

3580 SOUTH 144TH STREET

401 N FAIRVIEW AVE

499 W ELM STREET

1135 NM 554


Common street-address words include:

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


State-route style addresses can also be valid physical addresses.

Example:

1135 NM 554


A line beginning with a street/building number is a strong clue
that it is delivery_address.


============================================================
ADDRESS 2
============================================================

Lines within the Ship To block that are NOT:

- the primary organization name
- the physical street address
- city/state/ZIP

may belong in delivery_address2.


These can include:

- project name
- site name
- recipient name
- person's name
- attention line
- building name
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


Do NOT discard MEAGAN BIRDSALL.

Do NOT mistake QUEENSLAND MANOR for the street address.


============================================================
DELIVERY CONTACT
============================================================

delivery_contact should normally come from the printed field
specifically labeled:

Delivery contact

or another clearly identified delivery-contact field.

Do NOT automatically use a person's name from the Ship To block
as delivery_contact.


Example:

Contact phone       630-897-8489
Delivery contact    616-566-7124

Correct:

delivery_contact = "616-566-7124"


If the Delivery contact field includes both a phone number and
a person's name, preserve both when clearly visible.


============================================================
HANDLING UNITS
============================================================

Find EVERY genuine handwritten pallet / handling-unit notation.

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
DIMENSION ORDER
============================================================

Dimensions normally appear as:

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


These values are contextual clues only.

The visible handwriting is the primary source.

Do NOT force a value simply because it is common.


============================================================
NUMBER ROLE CHECK
============================================================

Do NOT assume every number in a handwritten line is a dimension.

Determine whether each visible number represents:

- length
- width
- height
- weight
- location
- quantity
- unrelated notation


Example:

48 x 44 / 756# / C27

should be interpreted approximately as:

length = 48
width = 44
height = null
weight = 756
location = C27


It should NOT become:

length = 48
width = 44
height = 756


============================================================
DO NOT SPLIT NUMBERS
============================================================

Do not split one handwritten number into two values unless the
visual handwriting clearly shows two distinct numbers.

Example:

756

should NOT become:

75
6


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

- return the most likely visual interpretation
- set uncertain = true
- explain the ambiguity briefly in notes


Do not force 98 simply because 98 is a common length.

Do not force 78 simply because the pen stroke resembles a 7.

Mark uncertainty when appropriate.


============================================================
WEIGHT
============================================================

The # symbol commonly means pounds.

Examples:

88# = 88 pounds

294# = 294 pounds

756# = 756 pounds


A large number associated with # is usually weight rather than
a dimension.


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
D20E
C21-2
A19-E


============================================================
PRINTED PRODUCT DIMENSIONS
============================================================

Do NOT use printed:

- product measurements
- catalog dimensions
- product-description measurements
- line-item measurements

as handling-unit dimensions.


Handling-unit dimensions should come from the handwritten
shipping notation unless the document clearly identifies another
actual handling-unit measurement.


============================================================
DO NOT CREATE FAKE HANDLING UNITS
============================================================

Do not create a handling unit from:

- initials
- signatures
- check marks
- lone locations
- isolated numbers
- miscellaneous handwriting
- packed quantities
- product measurements


A genuine HU should contain meaningful pallet/shipping structure.


============================================================
INCOMPLETE HANDLING UNIT
============================================================

A real handling unit may still contain one unreadable field.

Example:

98 x 45 x ? / 294# / C21-2


Return:

length = 98
width = 45
height = null
weight = 294
location = C21-2
uncertain = true


Do NOT discard the entire HU merely because one field cannot
be read.


============================================================
FINAL REVIEW — REQUIRED
============================================================

Before responding:

1. Locate the literal printed "Sales order" field.
2. Transcribe its value if readable.
3. Confirm the Sales Order was NOT created from SRP_number.
4. Confirm SRP_number remains a separate identifier.
5. Read the complete Ship To block.
6. Identify the actual physical street-address line.
7. Preserve project/person lines in Address 2 when appropriate.
8. Re-check every handwritten handling unit.
9. Confirm weights were not interpreted as dimensions.
10. Confirm printed product dimensions were ignored.
"""


# ==========================================================
# BASIC STRING HELPERS
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
    # LARGE "HEIGHT" THAT IS REALLY A WEIGHT
    #
    # Example:
    #
    # 48 x 44 x 756
    #
    # where 756 is actually pounds.
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
    # Example:
    #
    # 756 incorrectly interpreted as:
    #
    # height = 75
    # weight = 6
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
    # CONTRADICTORY HEIGHT
    #
    # Example:
    #
    # height = 75
    # weight = 756
    #
    # while the model's notes explicitly say the height
    # was unclear/not visible.
    #
    # Preserve obvious weight and clear questionable height.
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
                    f"because the model identified the height "
                    f"as unclear while weight {weight} was "
                    f"independently identified."
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

    # ------------------------------------------------------
    # UNUSUALLY TALL HEIGHT
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

    # Strong complete HU.
    if dimension_count >= 3:
        return True

    # Partial dimensions plus weight.
    if (
        dimension_count >= 2
        and has_weight
    ):
        return True

    # Partial dimensions plus warehouse location.
    if (
        dimension_count >= 2
        and has_location
        and _partial_dimensions_are_plausible(
            hu
        )
    ):
        return True

    # One dimension + weight + location.
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

        uploaded_file = client.files.create(
            file=pdf_file,
            purpose="user_data",
        )

        uploaded_file_id = uploaded_file.id

        response = client.responses.create(

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

        # --------------------------------------------------
        # MASTER SO VALIDATION
        #
        # GPT must have actually returned SO-########.
        # Anything else becomes null.
        # --------------------------------------------------

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

        # --------------------------------------------------
        # CLEAN HANDLING UNITS
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

        # --------------------------------------------------
        # USAGE
        # --------------------------------------------------

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

    # First real value wins.
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

    # ------------------------------------------------------
    # TWO DIFFERENT KNOWN MASTER SALES ORDERS
    #
    # Hard boundary.
    # ------------------------------------------------------

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

    # Strong matching identity means same shipment.
    if similarity >= 8:
        return False

    # ------------------------------------------------------
    # AN SO APPEARS AFTER UNIDENTIFIED PAGES
    #
    # Start temporary new group. Backward stitching may
    # reconnect them.
    # ------------------------------------------------------

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

    # ======================================================
    # ONE PDF PAGE PER GPT CALL
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

        # --------------------------------------------------
        # STRICT MASTER SO VALIDATION
        # --------------------------------------------------

        page_result[
            "sales_order"
        ] = _validate_sales_order(
            page_result.get(
                "sales_order"
            )
        )

        page_results.append(
            page_result
        )

        # --------------------------------------------------
        # TOKEN USAGE
        # --------------------------------------------------

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
        # DEBUG OUTPUT
        # --------------------------------------------------

        page_debug.append({

            "page":
                page_number,

            "sales_order":
                page_result.get(
                    "sales_order"
                ),

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
