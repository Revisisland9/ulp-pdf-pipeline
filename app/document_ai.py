import io
from typing import Any, Dict, List

from google.cloud import documentai
from pypdf import PdfReader, PdfWriter


# ==========================================================
# GOOGLE DOCUMENT AI CONFIG
# ==========================================================

PROJECT_ID = "706802237280"
LOCATION = "us"

# Existing PRODUCTION Custom Extractor.
#
# DO NOT CHANGE.
CUSTOM_PROCESSOR_ID = "108af97110febd0f"

# New Enterprise Document OCR processor.
#
# This will be used by the GPT hybrid test path.
PINK_OCR_PROCESSOR_ID = "711477af4e3c321d"


# Stay comfortably below Document AI's 15-page
# synchronous processing limit.
CHUNK_SIZE = 12

# Overlap adjacent chunks for the existing Custom Extractor
# so Sales Orders/entities near boundaries can be processed
# in both chunks.
OVERLAP_PAGES = 2


# ==========================================================
# CLIENT
# ==========================================================

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


# ==========================================================
# PROCESSOR RESOURCE NAMES
# ==========================================================

def _processor_name(
    client,
    processor_id: str,
):
    """
    Return the full Document AI processor resource name
    for the requested processor.
    """

    return client.processor_path(
        PROJECT_ID,
        LOCATION,
        processor_id,
    )


# ==========================================================
# PDF HELPERS
# ==========================================================

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


# ==========================================================
# GENERIC DOCUMENT AI CALL
# ==========================================================

def _process_pdf_bytes(
    client,
    processor_name: str,
    pdf_bytes: bytes,
):
    """
    Send one <=15-page PDF chunk to Document AI.

    This works for both:

    - Custom Extractor
    - Enterprise OCR
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


# ==========================================================
# CUSTOM EXTRACTOR ENTITY HANDLING
# ==========================================================

def _entity_from_document_ai(
    entity,
    original_start_page: int,
) -> Dict[str, Any]:
    """
    Convert a Custom Extractor Document AI entity into
    our flat entity format and restore its ORIGINAL
    PDF page number.

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


# ==========================================================
# OCR TEXT HELPERS
# ==========================================================

def _text_from_anchor(
    document,
    text_anchor,
) -> str:
    """
    Extract text represented by a Document AI text anchor.
    """

    if not text_anchor:
        return ""

    text_segments = (
        text_anchor.text_segments
        or []
    )

    parts = []

    for segment in text_segments:

        start_index = int(
            segment.start_index
            or 0
        )

        end_index = int(
            segment.end_index
            or 0
        )

        if end_index <= start_index:
            continue

        parts.append(
            document.text[
                start_index:end_index
            ]
        )

    return "".join(
        parts
    ).strip()


def _ocr_pages_from_document(
    document,
    original_start_page: int,
) -> List[Dict[str, Any]]:
    """
    Convert Enterprise OCR output into page-level text.

    Example result:

    [
        {
            "page": 1,
            "text": "..."
        },
        {
            "page": 2,
            "text": "..."
        }
    ]

    This gives the hybrid GPT path deterministic printed OCR
    organized by the ORIGINAL PDF page.
    """

    pages = []

    for local_page_index, page in enumerate(
        document.pages
    ):

        original_page_number = (
            original_start_page
            + local_page_index
            + 1
        )

        page_text = _text_from_anchor(
            document,
            page.layout.text_anchor,
        )

        pages.append({
            "page":
                original_page_number,

            "text":
                page_text,
        })

    return pages


# ==========================================================
# EXISTING PRODUCTION CUSTOM EXTRACTOR
# ==========================================================

def extract_ulp_document(
    pdf_bytes: bytes
):
    """
    EXISTING PRODUCTION ULP Document AI extraction.

    Uses:

        Custom Extractor
        108af97110febd0f

    Behavior remains the same as before.

    1. Count PDF pages.

    2. If <= 12 pages:
       process normally.

    3. If > 12 pages:
       split internally into overlapping chunks.

    4. Restore original PDF page numbers.

    5. Deduplicate entities created by overlap.

    6. Return one combined entity list to the existing
       ULP normalizer.

    Production callers do not need to change.
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
        client,
        CUSTOM_PROCESSOR_ID,
    )

    all_entities = []

    text_parts = []

    # ======================================================
    # SMALL PDF
    # ======================================================

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

    # ======================================================
    # LARGE PDF
    # ======================================================

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

        text_parts.append(
            (
                f"\n"
                f"===== CHUNK {chunk_number} "
                f"ORIGINAL PAGES "
                f"{start_page + 1}-{end_page} =====\n"
                f"{document.text}"
            )
        )

        for entity in document.entities:

            all_entities.append(
                _entity_from_document_ai(
                    entity,
                    original_start_page=start_page,
                )
            )

        if end_page >= total_pages:
            break

        start_page += step

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


# ==========================================================
# NEW ENTERPRISE OCR PATH
# ==========================================================

def extract_pink_ocr(
    pdf_bytes: bytes
):
    """
    NEW Enterprise OCR extraction for the GPT hybrid path.

    Uses:

        ULP PINK OCR
        711477af4e3c321d

    This does NOT use or alter the existing Custom Extractor.

    Its job is to provide reliable PRINTED OCR text.

    Output:

    {
        "text": "...",
        "pages": [
            {
                "page": 1,
                "text": "..."
            },
            ...
        ],
        "page_count": 15,
        "chunk_count": 2
    }

    We intentionally do NOT expect Custom Extractor entities
    from Enterprise OCR.

    GPT/Python will later use this page-level printed text to
    recover fields such as:

        Sales order
        Customer PO
        SRP
        delivery/contact information
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
        client,
        PINK_OCR_PROCESSOR_ID,
    )

    all_pages = []

    text_parts = []

    # ======================================================
    # SMALL PDF
    # ======================================================

    if total_pages <= CHUNK_SIZE:

        result = _process_pdf_bytes(
            client,
            processor_name,
            pdf_bytes,
        )

        document = (
            result.document
        )

        all_pages.extend(
            _ocr_pages_from_document(
                document,
                original_start_page=0,
            )
        )

        return {
            "text":
                document.text,

            "pages":
                all_pages,

            "page_count":
                total_pages,

            "chunk_count":
                1,
        }

    # ======================================================
    # LARGE PDF
    #
    # Enterprise OCR does NOT need overlap because we are
    # collecting ordinary page-level OCR rather than
    # cross-page Custom Extractor entities.
    # ======================================================

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

        text_parts.append(
            (
                f"\n"
                f"===== OCR CHUNK {chunk_number} "
                f"ORIGINAL PAGES "
                f"{start_page + 1}-{end_page} =====\n"
                f"{document.text}"
            )
        )

        all_pages.extend(
            _ocr_pages_from_document(
                document,
                original_start_page=start_page,
            )
        )

        if end_page >= total_pages:
            break

        # No overlap needed for basic OCR.
        start_page = end_page

    all_pages.sort(
        key=lambda p:
            p.get(
                "page"
            )
            or 999999
    )

    return {
        "text":
            "\n".join(
                text_parts
            ),

        "pages":
            all_pages,

        "page_count":
            total_pages,

        "chunk_count":
            chunk_number,
    }
