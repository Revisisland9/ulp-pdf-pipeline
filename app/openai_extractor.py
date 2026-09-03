import os
import json
from io import BytesIO

from openai import OpenAI


MODEL = "gpt-5.4-mini"


def _get_client():
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured."
        )

    return OpenAI(
        api_key=api_key
    )


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


EXTRACTION_INSTRUCTIONS = """
You are extracting shipping information from scanned ULP "Pink"
shipping documents.

Carefully inspect the actual visual content of every PDF page,
including handwriting.

Return every Sales Order and every handwritten handling unit.

IMPORTANT RULES

SALES ORDERS
- A packet may contain multiple Sales Orders.
- Group pages belonging to the same Sales Order together.
- Preserve original PDF page numbers, starting with page 1.
- Sales Order values commonly begin with SO-.
- Do not combine different Sales Orders.

SHIPMENT FIELDS
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

Customer reference / requisition / PO information should be placed
in customer_PO when it represents the customer's purchase order.

HANDLING UNITS
This is the most important part of the extraction.

Find EVERY handwritten handling-unit or pallet notation.

There may be:
- one handling unit
- multiple handling units on the same page
- numbered handwritten lines such as 1, 2, 3, etc.

For every handling unit extract:

- length
- width
- height
- weight
- location
- page

Typical handwritten notation may resemble:

48 x 45 x 26 / 97# / B8

or

1 - 74 x 45 x 55 - 542# - D24

The "#" symbol commonly means pounds / weight.

DIMENSION RULES
- Dimensions are generally written L x W x H.
- The first number is normally length.
- The second number is normally width.
- The third number is normally height.
- Common shipment lengths include:
  48, 72, 74, 79, 96, 98, 120, 144.
- If handwriting clearly shows two 48 dimensions plus one other
  dimension, the intended orientation may be 48 x 48 x other.
- Do not blindly change what is visibly written merely to match a
  common dimension.
- Use the visual evidence first.

VERY IMPORTANT
- Do NOT use dimensions from printed item/product tables as pallet
  dimensions.
- Concentrate on handwritten shipping annotations.
- Do not miss a handling unit simply because handwriting is faint,
  cramped, slanted, or overlaps printed material.
- Re-scan each page visually for additional numbered handwritten
  handling-unit lines before finishing.

LOCATION
- Location is normally a short handwritten warehouse/location code.
- Examples could resemble B8, D24, D22E, 07, 09.
- Do not append unrelated handwritten numbers to the location.
- If unsure whether trailing handwriting belongs to the location,
  return the most likely location and set uncertain=true.

UNCERTAINTY
- Never invent a value.
- If a field cannot be read, return null.
- Set uncertain=true if ANY part of a handling unit is genuinely
  ambiguous.
- Use notes to briefly describe the ambiguity.
- Do not manufacture numeric confidence percentages.

ADDRESS RULES
- Preserve Address 2 separately when it is clearly present.
- An ATTN line may belong in delivery_address2 rather than
  delivery_name when visually appropriate.
- Do not invent an Address 2 merely because one might exist elsewhere.

FINAL COMPLETENESS CHECK
Before returning the answer:
1. Re-scan every page.
2. Count visibly handwritten handling-unit lines.
3. Verify that every visible handling-unit line is represented.
4. Confirm printed product/item dimensions were not mistakenly used.
5. Confirm handling units are attached to the correct Sales Order.
"""


def extract_ulp_with_gpt(
    pdf_bytes: bytes
):
    """
    Extract a ULP Pink PDF using OpenAI vision/file understanding.

    Current test behavior:
      PDF
        -> OpenAI file upload
        -> GPT visual document extraction
        -> strict structured JSON
        -> returned to /api/v1/ulp/extract-gpt

    This does NOT yet feed the existing sheet mapper.
    We first want to inspect GPT's raw structured result.
    """

    if not pdf_bytes:
        raise ValueError(
            "PDF is empty."
        )

    client = _get_client()

    uploaded_file_id = None

    try:

        #
        # Upload the PDF temporarily.
        #
        # user_data is the flexible file purpose intended
        # for files supplied to model requests.
        #
        pdf_file = BytesIO(
            pdf_bytes
        )

        pdf_file.name = "ulp_pink.pdf"

        uploaded_file = client.files.create(
            file=pdf_file,
            purpose="user_data",
        )

        uploaded_file_id = (
            uploaded_file.id
        )

        #
        # Ask GPT to visually inspect the PDF and return
        # only our defined extraction structure.
        #
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
                        "ulp_pink_extraction",

                    "strict":
                        True,

                    "schema":
                        ULP_SCHEMA,
                }
            },
        )

        #
        # Structured Outputs still arrive as JSON text.
        #
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

        #
        # Return extraction plus usage information.
        #
        # Usage is extremely useful during testing because
        # it will let us determine the REAL cost of these
        # Pink packets.
        #
        usage = None

        if response.usage:
            usage = {
                "input_tokens":
                    response.usage.input_tokens,

                "output_tokens":
                    response.usage.output_tokens,

                "total_tokens":
                    response.usage.total_tokens,
            }

        return {
            "ok":
                True,

            "model":
                MODEL,

            "extraction":
                extracted,

            "usage":
                usage,
        }

    finally:

        #
        # Do not leave every Pink PDF sitting in OpenAI
        # file storage after processing.
        #
        if uploaded_file_id:

            try:
                client.files.delete(
                    uploaded_file_id
                )

            except Exception:
                #
                # File cleanup failure should not cause a
                # successful shipment extraction to fail.
                #
                pass
