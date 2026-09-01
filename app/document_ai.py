import io
from typing import Any, Dict, List

from google.cloud import documentai
from pypdf import PdfReader, PdfWriter


PROJECT_ID = "706802237280"
LOCATION = "us"
PROCESSOR_ID = "108af97110febd0f"


# Stay comfortably below Document AI's 15-page
# synchronous processing limit.
CHUNK_SIZE = 12

# Overlap adjacent chunks so Sales Orders near a
# boundary have pages processed in both chunks.
OVERLAP_PAGES = 2


def _get_client():
    """
    Create the Google Document AI client.
    """

    return documentai.DocumentProcessorServiceClient(
        client_options={
            "api_endpoint":
                f"{LOCATION}-documentai.googleapis.com"
        }
    )


def _processor_name(client):
    """
    Return the full Document AI processor resource name.
    """

    return client.processor_path(
        PROJECT_ID,
        LOCATION,
        PROCESSOR_ID,
    )


def _pdf_page_count(
    pdf_bytes: bytes
) -> int:
    """
    Count pages in the incoming PDF.
    """

    reader = PdfReader(
        io.BytesIO(pdf_bytes)
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
    Create a smaller PDF containing:

        start_page through end_page

    Page indexes here are zero-based and end_page
    is exclusive.

    Example:
        start_page = 0
        end_page   = 12

    creates original pages 1-12.
    """

    reader = PdfReader(
        io.BytesIO(pdf_bytes)
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

    output = io.BytesIO()

    writer.write(
        output
    )

    return output.getvalue()


def _process_pdf_bytes(
    client,
    processor_name: str,
    pdf_bytes: bytes,
):
    """
    Send one <=15-page PDF chunk to Document AI.
    """

    raw_document = documentai.RawDocument(
        content=pdf_bytes,
        mime_type="application/pdf",
    )

    request = documentai.ProcessRequest(
        name=processor_name,
        raw_document=raw_document,
    )

    return client.process_document(
        request=request
    )


def _entity_from_document_ai(
    entity,
    original_start_page: int,
) -> Dict[str, Any]:
    """
    Convert a Document AI entity into our flat entity
    format and restore its ORIGINAL PDF page number.

    Document AI page indexes are local to each chunk.

    Example:

        chunk starts at original page 11
        Document AI says local page = 1

    That means original PDF page 12.
    """

    page = None

    if entity.page_anchor.page_refs:

        page_ref = (
            entity.page_anchor.page_refs[0]
        )

        if page_ref.page is not None:

            local_page = int(
                page_ref.page
            )

            page = (
                original_start_page
                + local_page
                + 1
            )

    return {
        "type":
            entity.type_,

        "value":
            entity.mention_text,

        "confidence":
            float(
                entity.confidence
            ),

        "page":
            page,
    }


def _deduplicate_entities(
    entities: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Remove duplicate entities created by overlapping chunks.

    A duplicate means:

        same entity type
        same extracted value
        same ORIGINAL PDF page

    If the same entity was extracted twice with different
    confidence scores, keep the higher-confidence version.
    """

    best = {}

    for entity in entities:

        entity_type = str(
            entity.get("type") or ""
        ).strip()

        value = str(
            entity.get("value") or ""
        ).strip()

        page = entity.get(
            "page"
        )

        key = (
            entity_type,
            value,
            page,
        )

        existing = best.get(
            key
        )

        if existing is None:

            best[key] = entity
            continue

        new_confidence = float(
            entity.get("confidence")
            or 0
        )

        old_confidence = float(
            existing.get("confidence")
            or 0
        )

        if new_confidence > old_confidence:
            best[key] = entity

    deduped = list(
        best.values()
    )

    #
    # Keep output in document order.
    #
    deduped.sort(
        key=lambda e: (
            e.get("page")
            if e.get("page") is not None
            else 999999,

            str(
                e.get("type") or ""
            ),

            str(
                e.get("value") or ""
            ),
        )
    )

    return deduped


def extract_ulp_document(
    pdf_bytes: bytes
):
    """
    Main ULP Document AI extraction function.

    Behavior:

    1. Count PDF pages.

    2. If <= 12 pages:
       process normally.

    3. If > 12 pages:
       split internally into overlapping chunks.

       Example 60-page PDF:

           chunk 1 = pages 1-12
           chunk 2 = pages 11-22
           chunk 3 = pages 21-32
           chunk 4 = pages 31-42
           chunk 5 = pages 41-52
           chunk 6 = pages 51-60

    4. Restore original PDF page numbers.

    5. Deduplicate entities created by overlap.

    6. Return one combined entity list to the existing
       ULP normalizer.

    The rest of the application does not need to know
    that the PDF was split.
    """

    if not pdf_bytes:
        raise ValueError(
            "PDF is empty."
        )

    total_pages = _pdf_page_count(
        pdf_bytes
    )

    if total_pages == 0:
        raise ValueError(
            "PDF contains no pages."
        )

    client = _get_client()

    processor_name = _processor_name(
        client
    )

    all_entities = []

    text_parts = []

    #
    # SMALL PDF
    #
    # No splitting necessary.
    #
    if total_pages <= CHUNK_SIZE:

        result = _process_pdf_bytes(
            client,
            processor_name,
            pdf_bytes,
        )

        document = (
            result.document
        )

        for entity in document.entities:

            all_entities.append(
                _entity_from_document_ai(
                    entity,
                    original_start_page=0,
                )
            )

        return {
            "text":
                document.text,

            "entities":
                all_entities,

            "page_count":
                total_pages,

            "chunk_count":
                1,
        }

    #
    # LARGE PDF
    #
    # Split into overlapping chunks.
    #
    step = (
        CHUNK_SIZE
        - OVERLAP_PAGES
    )

    start_page = 0

    chunk_number = 0

    while start_page < total_pages:

        end_page = min(
            start_page + CHUNK_SIZE,
            total_pages,
        )

        chunk_number += 1

        chunk_pdf = _make_pdf_chunk(
            pdf_bytes,
            start_page,
            end_page,
        )

        result = _process_pdf_bytes(
            client,
            processor_name,
            chunk_pdf,
        )

        document = (
            result.document
        )

        #
        # Store OCR text mainly for debugging.
        #
        text_parts.append(
            (
                f"\n"
                f"===== CHUNK {chunk_number} "
                f"ORIGINAL PAGES "
                f"{start_page + 1}-{end_page} =====\n"
                f"{document.text}"
            )
        )

        #
        # Restore every entity to the page number
        # from the ORIGINAL 60-page PDF.
        #
        for entity in document.entities:

            all_entities.append(
                _entity_from_document_ai(
                    entity,
                    original_start_page=start_page,
                )
            )

        #
        # Finished.
        #
        if end_page >= total_pages:
            break

        start_page += step

    #
    # Because chunks overlap, pages 11-12, 21-22,
    # etc. may have been extracted twice.
    #
    all_entities = (
        _deduplicate_entities(
            all_entities
        )
    )

    return {
        "text":
            "\n".join(
                text_parts
            ),

        "entities":
            all_entities,

        "page_count":
            total_pages,

        "chunk_count":
            chunk_number,
    }
