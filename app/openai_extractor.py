import os
from openai import OpenAI


client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"]
)


def extract_ulp_with_gpt(pdf_bytes: bytes):
    """
    Temporary GPT test extractor.

    For now this only verifies that:
    - Cloud Run can read OPENAI_API_KEY
    - the OpenAI SDK works
    - the backend can successfully call OpenAI

    We will add the actual PDF/image extraction logic next.
    """

    response = client.responses.create(
        model="gpt-5.4-mini",
        input="Reply with exactly: OPENAI CONNECTION WORKING"
    )

    return {
        "ok": True,
        "message": response.output_text
    }
