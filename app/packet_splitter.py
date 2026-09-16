import io
import re
import zipfile

from typing import (
    Any,
    Dict,
    List,
    Optional,
)

from pypdf import (
    PdfReader,
    PdfWriter,
)

from app.document_ai import (
    extract_pink_ocr,
)


# ==========================================================
# SALES ORDER RULES
# ==========================================================

SALES_ORDER_PATTERN = re.compile(
    r"^SO-\d{8}$",
    re.IGNORECASE,
)

SALES_ORDER_SEARCH_PATTERN = re.compile(
    r"\bSO-\d{8}\b",
    re.IGNORECASE,
)


# ==========================================================
# SALES ORDER HELPERS
# ==========================================================

def _validate_sales_order(
    value: Any,
) -> Optional[str]:

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


def _extract_sales_order_from_ocr(
    page_text: str,
) -> Optional[str]:

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
    # LOOK NEAR SALES ORDER LABEL
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
    # UNIQUE SALES ORDER ANYWHERE ON PAGE
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


def _sales_order_sort_key(
    sales_order: str,
) -> int:
    """
    SO-00325812 -> 325812
    """

    digits = re.sub(
        r"\D",
        "",
        sales_order
        or "",
    )

    if not digits:
        return 0

    return int(
        digits
    )


# ==========================================================
# OCR PAGE MAP
# ==========================================================

def _build_ocr_page_map(
    ocr_result: Dict[str, Any],
) -> Dict[int, str]:

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
# DETECT SALES ORDERS BY PAGE
# ==========================================================

def detect_sales_order_pages(
    pdf_bytes: bytes,
) -> Dict[str, Any]:

    if not pdf_bytes:

        raise ValueError(
            "PDF is empty."
        )

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

    ocr_page_map = (
        _build_ocr_page_map(
            ocr_result
        )
    )

    pages = []

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

        pages.append({

            "page":
                page_number,

            "sales_order":
                sales_order,

            "ocr_text_length":
                len(
                    page_text
                ),
        })

    return {

        "ok":
            True,

        "page_count":
            total_pages,

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

        "pages":
            pages,
    }


# ==========================================================
# GROUP PAGES INTO PACKETS
# ==========================================================

def _group_pages_into_packets(
    page_results: List[
        Dict[str, Any]
    ],
) -> List[
    Dict[str, Any]
]:
    """
    Group pages sequentially.

    Rules:

    - First valid SO starts the first packet.
    - Same SO stays in current packet.
    - Null SO inherits current packet.
    - Different valid SO starts a new packet.
    - Pages before the first valid SO are NOT silently assigned.
    """

    packets = []

    current_packet = None

    for page in page_results:

        page_number = page.get(
            "page"
        )

        sales_order = (
            _validate_sales_order(
                page.get(
                    "sales_order"
                )
            )
        )

        # --------------------------------------------------
        # NO ACTIVE PACKET YET
        # --------------------------------------------------

        if current_packet is None:

            if not sales_order:

                packets.append({

                    "sales_order":
                        None,

                    "pages":
                        [
                            page_number
                        ],

                    "status":
                        "unassigned",
                })

                continue

            current_packet = {

                "sales_order":
                    sales_order,

                "pages":
                    [
                        page_number
                    ],

                "status":
                    "assigned",
            }

            continue

        # --------------------------------------------------
        # NULL PAGE
        # INHERIT CURRENT SALES ORDER
        # --------------------------------------------------

        if not sales_order:

            current_packet[
                "pages"
            ].append(
                page_number
            )

            continue

        # --------------------------------------------------
        # SAME SALES ORDER
        # --------------------------------------------------

        if (
            sales_order
            == current_packet[
                "sales_order"
            ]
        ):

            current_packet[
                "pages"
            ].append(
                page_number
            )

            continue

        # --------------------------------------------------
        # DIFFERENT SALES ORDER
        # CLOSE CURRENT PACKET AND START NEXT
        # --------------------------------------------------

        packets.append(
            current_packet
        )

        current_packet = {

            "sales_order":
                sales_order,

            "pages":
                [
                    page_number
                ],

            "status":
                "assigned",
        }

    if current_packet is not None:

        packets.append(
            current_packet
        )

    return packets


# ==========================================================
# VALIDATE PACKET ASSIGNMENT
# ==========================================================

def _validate_packet_assignment(
    packets: List[
        Dict[str, Any]
    ],
    total_pages: int,
) -> Dict[str, Any]:

    assigned_pages = []
    unassigned_pages = []

    for packet in packets:

        sales_order = packet.get(
            "sales_order"
        )

        pages = (
            packet.get(
                "pages"
            )
            or []
        )

        if sales_order:

            assigned_pages.extend(
                pages
            )

        else:

            unassigned_pages.extend(
                pages
            )

    duplicates = []

    seen = set()

    for page_number in assigned_pages:

        if page_number in seen:

            duplicates.append(
                page_number
            )

        seen.add(
            page_number
        )

    expected_pages = set(
        range(
            1,
            total_pages + 1,
        )
    )

    actual_pages = set(
        assigned_pages
        + unassigned_pages
    )

    missing_pages = sorted(
        expected_pages
        - actual_pages
    )

    unexpected_pages = sorted(
        actual_pages
        - expected_pages
    )

    return {

        "assigned_pages":
            sorted(
                assigned_pages
            ),

        "assigned_page_count":
            len(
                assigned_pages
            ),

        "unassigned_pages":
            sorted(
                unassigned_pages
            ),

        "duplicate_pages":
            sorted(
                set(
                    duplicates
                )
            ),

        "missing_pages":
            missing_pages,

        "unexpected_pages":
            unexpected_pages,

        "valid":
            (
                len(
                    unassigned_pages
                )
                == 0

                and len(
                    duplicates
                )
                == 0

                and len(
                    missing_pages
                )
                == 0

                and len(
                    unexpected_pages
                )
                == 0

                and len(
                    assigned_pages
                )
                == total_pages
            ),
    }


# ==========================================================
# BUILD ONE PDF
# ==========================================================

def _build_packet_pdf(
    reader: PdfReader,
    page_numbers: List[int],
) -> bytes:

    writer = PdfWriter()

    for page_number in page_numbers:

        page_index = (
            page_number
            - 1
        )

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
# SPLIT PDF + CREATE ZIP
# ==========================================================

def split_packet_to_zip(
    pdf_bytes: bytes,
) -> Dict[str, Any]:
    """
    Full packet splitter.

    Returns:

        {
            "zip_bytes": ...,
            "manifest": {...}
        }
    """

    if not pdf_bytes:

        raise ValueError(
            "PDF is empty."
        )

    reader = PdfReader(
        io.BytesIO(
            pdf_bytes
        )
    )

    total_pages = len(
        reader.pages
    )

    if total_pages <= 0:

        raise ValueError(
            "PDF contains no pages."
        )

    # ======================================================
    # OCR
    # ======================================================

    detection = (
        detect_sales_order_pages(
            pdf_bytes
        )
    )

    page_results = (
        detection.get(
            "pages"
        )
        or []
    )

    # ======================================================
    # GROUP
    # ======================================================

    packets = (
        _group_pages_into_packets(
            page_results
        )
    )

    # ======================================================
    # VALIDATE
    # ======================================================

    validation = (
        _validate_packet_assignment(
            packets,
            total_pages,
        )
    )

    if not validation[
        "valid"
    ]:

        raise ValueError(
            (
                "Packet assignment validation failed. "
                f"Unassigned={validation['unassigned_pages']}, "
                f"Duplicates={validation['duplicate_pages']}, "
                f"Missing={validation['missing_pages']}, "
                f"Unexpected={validation['unexpected_pages']}"
            )
        )

    # ======================================================
    # ONLY ASSIGNED PACKETS
    # ======================================================

    assigned_packets = [
        packet
        for packet in packets
        if packet.get(
            "sales_order"
        )
    ]

    # ======================================================
    # SORT BY NUMERIC SO
    # ======================================================

    assigned_packets.sort(
        key=lambda packet:
            _sales_order_sort_key(
                packet[
                    "sales_order"
                ]
            )
    )

    # ======================================================
    # BUILD ZIP
    # ======================================================

    zip_buffer = io.BytesIO()

    manifest_packets = []

    with zipfile.ZipFile(
        zip_buffer,
        mode="w",
        compression=
            zipfile.ZIP_DEFLATED,
    ) as zip_file:

        for packet in assigned_packets:

            sales_order = packet[
                "sales_order"
            ]

            page_numbers = packet[
                "pages"
            ]

            filename = (
                f"{sales_order}.pdf"
            )

            packet_pdf = (
                _build_packet_pdf(
                    reader,
                    page_numbers,
                )
            )

            zip_file.writestr(
                filename,
                packet_pdf,
            )

            manifest_packets.append({

                "sales_order":
                    sales_order,

                "filename":
                    filename,

                "pages":
                    page_numbers,

                "page_count":
                    len(
                        page_numbers
                    ),
            })

    zip_bytes = (
        zip_buffer.getvalue()
    )

    # ======================================================
    # FINAL MANIFEST
    # ======================================================

    manifest = {

        "ok":
            True,

        "workflow":
            "ulp_packet_split",

        "page_count":
            total_pages,

        "packet_count":
            len(
                manifest_packets
            ),

        "assigned_page_count":
            validation[
                "assigned_page_count"
            ],

        "unassigned_pages":
            validation[
                "unassigned_pages"
            ],

        "duplicate_pages":
            validation[
                "duplicate_pages"
            ],

        "missing_pages":
            validation[
                "missing_pages"
            ],

        "packets":
            manifest_packets,

        "google_ocr":
            detection.get(
                "google_ocr"
            ),
    }

    return {

        "zip_bytes":
            zip_bytes,

        "manifest":
            manifest,
    }
