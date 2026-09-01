import os
from google.cloud import documentai

PROJECT_ID = "706802237280"
LOCATION = "us"
PROCESSOR_ID = "108af97110febd0f"


def extract_ulp_document(pdf_bytes: bytes):
    client = documentai.DocumentProcessorServiceClient(
        client_options={
            "api_endpoint": f"{LOCATION}-documentai.googleapis.com"
        }
    )

    processor_name = client.processor_path(
        PROJECT_ID,
        LOCATION,
        PROCESSOR_ID
    )

    raw_document = documentai.RawDocument(
        content=pdf_bytes,
        mime_type="application/pdf"
    )

    request = documentai.ProcessRequest(
        name=processor_name,
        raw_document=raw_document
    )

    result = client.process_document(request=request)

    entities = []

    for entity in result.document.entities:
        page = None

        if entity.page_anchor.page_refs:
            page_ref = entity.page_anchor.page_refs[0]
            if page_ref.page is not None:
                page = int(page_ref.page) + 1

        entities.append({
            "type": entity.type_,
            "value": entity.mention_text,
            "confidence": entity.confidence,
            "page": page
        })

    return {
        "text": result.document.text,
        "entities": entities
    }
