import re

from typing import (
    Any,
    Dict,
    List,
    Optional,
)

from app.document_ai import (
    extract_pink_ocr,
)


# ==========================================================
# SALES ORDER RULES
# ==========================================================

# Master ULP Sales Order:
#
# SO-
# followed by exactly 8 digits
#
# Example:
# SO-00325428

SALES_ORDER_PATTERN = re.compile(
    r"^SO-\d{8}$",
    re.IGNORECASE,
)

SALES_ORDER_SEARCH_PATTERN = re.compile(
    r"\bSO-\d{8}\b",
    re.IGNORECASE,
)


# ==========================================================
# SALES ORDER VALIDATION
# ==========================================================

def _validate_sales_order(
    value: Any,
) -> Optional[str]:
    """
    Validate and normalize a ULP master Sales Order.

    Valid example:

        SO-00325428
    """

    if value is None:
        return None

    value = str(
        value
    ).strip().upper()

    if not value:
        return None

    if not SALES_ORDER_PATTERN.fullmatch(
        value
    ):
        return None

    return value


# ==========================================================
# FIND SALES ORDER IN OCR TEXT
# ==========================================================

def _extract_sales_order_from_ocr(
    page_text: str,
) -> Optional[str]:
    """
    Find a valid ULP master Sales Order in one page of
    Google Enterprise OCR text.

    Priority:

    1. Look near a printed "Sales Order" label.

    2. If that fails, look for all SO-######## values on
       the page.

    3. If exactly one unique valid Sales Order exists,
       accept it.

    4. If multiple different Sales Orders exist and we
       cannot confidently determine which one is the master
       Sales Order, return None.
    """

    page_text = (
        page_text
        or ""
    )

    if not page_text.strip():
        return None

    lines = [
        line.strip()
        for line in page_text.splitlines()
        if line.strip()
    ]

    # ======================================================
    # PASS 1
    # LOOK NEAR "SALES ORDER" LABEL
    # ======================================================

    for i, line in enumerate(
        lines
    ):

        normalized_label = re.sub(
            r"\s+",
            " ",
            line,
        ).strip().lower()

        compact = (
            normalized_label
            .replace(
                " ",
                ""
            )
        )

        if (
            "sales order"
            not in normalized_label

            and "salesorder"
            not in compact
        ):
            continue

        # Search the Sales Order label line and the next
        # few OCR lines.
        nearby = "\n".join(
            lines[
                i:min(
                    i + 4,
                    len(lines),
                )
            ]
        )

        match = (
            SALES_ORDER_SEARCH_PATTERN
            .search(
                nearby
            )
        )

        if match:

            return (
                _validate_sales_order(
                    match.group(0)
                )
            )

    # ======================================================
    # PASS 2
    # UNIQUE SO-######## ANYWHERE ON PAGE
    # ======================================================

    matches = (
        SALES_ORDER_SEARCH_PATTERN
        .findall(
            page_text
        )
    )

    unique = []

    seen = set()

    for match in matches:

        value = (
            match
            .upper()
            .strip()
        )

        if value in seen:
            continue

        seen.add(
            value
        )

        unique.append(
            value
        )

    if len(unique) == 1:

        return (
            _validate_sales_order(
                unique[0]
            )
        )

    return None


# ==========================================================
# BUILD OCR PAGE MAP
# ==========================================================

def _build_ocr_page_map(
    ocr_result: Dict[str, Any],
) -> Dict[int, str]:
    """
    Convert Document AI page output into:

        {
            1: "OCR text...",
            2: "OCR text...",
            ...
        }
    """

    page_map = {}

    for item in (
        ocr_result.get(
            "pages"
        )
        or []
    ):

        page_number = item.get(
            "page"
        )

        if page_number is None:
            continue

        page_map[
            int(
                page_number
            )
        ] = (
            item.get(
                "text"
            )
            or ""
        )

    return page_map


# ==========================================================
# DETECT PACKET STARTS
# ==========================================================

def detect_sales_order_pages(
    pdf_bytes: bytes,
) -> Dict[str, Any]:
    """
    Analyze an entire scanned packet using the EXISTING
    Google Enterprise OCR processor.

    This function DOES NOT split the PDF yet.

    Current purpose:

        PDF
            ↓
        Google Enterprise OCR
            ↓
        OCR text organized by original page number
            ↓
        detect valid SO-######## values
            ↓
        return packet-start candidates

    A page containing a confidently identified master
    Sales Order is treated as a packet-start candidate.

    Later versions will use these boundaries to create
    the individual PDFs.
    """

    if not pdf_bytes:

        raise ValueError(
            "PDF is empty."
        )

    # ======================================================
    # GOOGLE ENTERPRISE OCR
    # ======================================================

    ocr_result = (
        extract_pink_ocr(
            pdf_bytes
        )
    )

    total_pages = (
        ocr_result.get(
            "page_count"
        )
        or 0
    )

    if total_pages <= 0:

        raise ValueError(
            "OCR returned no PDF pages."
        )

    # ======================================================
    # PAGE MAP
    # ======================================================

    ocr_page_map = (
        _build_ocr_page_map(
            ocr_result
        )
    )

    # ======================================================
    # DETECT SALES ORDERS
    # ======================================================

    packet_starts: List[
        Dict[str, Any]
    ] = []

    page_diagnostics: List[
        Dict[str, Any]
    ] = []

    for page_number in range(
        1,
        total_pages + 1,
    ):

        page_text = (
            ocr_page_map.get(
                page_number,
                "",
            )
        )

        sales_order = (
            _extract_sales_order_from_ocr(
                page_text
            )
        )

        page_diagnostics.append({

            "page":
                page_number,

            "sales_order":
                sales_order,

            "ocr_text_length":
                len(
                    page_text
                ),
        })

        if sales_order:

            packet_starts.append({

                "page":
                    page_number,

                "sales_order":
                    sales_order,
            })

    # ======================================================
    # RETURN
    # ======================================================

    return {

        "ok":
            True,

        "page_count":
            total_pages,

        "packet_start_count":
            len(
                packet_starts
            ),

        "packet_starts":
            packet_starts,

        "google_ocr": {

            "processor":
                "711477af4e3c321d",

            "page_count":
                ocr_result.get(
                    "page_count"
                ),

            "chunk_count":
                ocr_result.get(
                    "chunk_count"
                ),
        },

        # Diagnostic only.
        # Useful while developing against new packet formats.
        "pages":
            page_diagnostics,
    }
