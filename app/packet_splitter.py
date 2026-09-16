import io
import re
import json
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
    # LOOK NEAR "SALES ORDER"
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
    # UNIQUE SO ANYWHERE ON PAGE
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

    digits = re.sub(
        r"\D",
        "",
        sales_order
        or "",
    )

    return (
        int(digits)
        if digits
        else 0
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
            "OCR returned no pages."
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

        "page_count":
            total_pages,

        "pages":
            pages,

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
    }


# ==========================================================
# GROUP PAGES INTO SEQUENTIAL PACKETS
# ==========================================================

def _group_pages_into_packets(
    page_results: List[
        Dict[str, Any]
    ],
) -> List[
    Dict[str, Any]
]:
    """
    Rules:

    - A valid SO starts a packet.
    - Repeated same SO stays in packet.
    - Null SO inherits current packet.
    - Different valid SO starts next packet.
    - Pages before the first detected SO remain unassigned.
    """

    packets = []

    current_packet = None

    leading_unassigned = []

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
        # BEFORE FIRST VALID SO
        # --------------------------------------------------

        if current_packet is None:

            if not sales_order:

                leading_unassigned.append(
                    page_number
                )

                continue

            if leading_unassigned:

                packets.append({

                    "sales_order":
                        None,

                    "pages":
                        leading_unassigned,

                    "status":
                        "unassigned",
                })

                leading_unassigned = []

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
        # NULL SO
        # INHERIT ACTIVE PACKET
        # --------------------------------------------------

        if not sales_order:

            current_packet[
                "pages"
            ].append(
                page_number
            )

            continue

        # --------------------------------------------------
        # SAME SO
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
        # NEW SO
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

    # ------------------------------------------------------
    # FINISH ACTIVE PACKET
    # ------------------------------------------------------

    if current_packet is not None:

        packets.append(
            current_packet
        )

    # ------------------------------------------------------
    # PDF WITH NO DETECTED SO AT ALL
    # ------------------------------------------------------

    elif leading_unassigned:

        packets.append({

            "sales_order":
                None,

            "pages":
                leading_unassigned,

            "status":
                "unassigned",
        })

    return packets


# ==========================================================
# FIND NON-CONTIGUOUS DUPLICATE SALES ORDERS
# ==========================================================

def _find_noncontiguous_duplicate_sales_orders(
    packets: List[
        Dict[str, Any]
    ],
) -> List[str]:

    counts = {}

    for packet in packets:

        sales_order = (
            _validate_sales_order(
                packet.get(
                    "sales_order"
                )
            )
        )

        if not sales_order:
            continue

        counts[
            sales_order
        ] = (
            counts.get(
                sales_order,
                0,
            )
            + 1
        )

    duplicates = [
        sales_order
        for sales_order, count
        in counts.items()
        if count > 1
    ]

    return sorted(
        duplicates,
        key=_sales_order_sort_key,
    )


# ==========================================================
# BUILD ANALYSIS / REVIEW PLAN
# ==========================================================

def analyze_packet(
    pdf_bytes: bytes,
) -> Dict[str, Any]:
    """
    Analyze the packet and determine:

    - valid SO packets
    - pages requiring review
    - exceptions
    - final status

    Every source page should ultimately be represented either
    by a valid SO PDF or REVIEW_REQUIRED_PAGES.pdf.
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
    # OCR + SO DETECTION
    # ======================================================

    detection = (
        detect_sales_order_pages(
            pdf_bytes
        )
    )

    detected_page_count = (
        detection.get(
            "page_count"
        )
        or 0
    )

    if detected_page_count != total_pages:

        raise ValueError(
            (
                "OCR page count does not match PDF page count. "
                f"PDF={total_pages}, "
                f"OCR={detected_page_count}"
            )
        )

    page_results = (
        detection.get(
            "pages"
        )
        or []
    )

    # ======================================================
    # INITIAL SEQUENTIAL GROUPING
    # ======================================================

    raw_packets = (
        _group_pages_into_packets(
            page_results
        )
    )

    duplicate_sos = (
        _find_noncontiguous_duplicate_sales_orders(
            raw_packets
        )
    )

    duplicate_so_set = set(
        duplicate_sos
    )

    valid_packets = []

    review_pages = set()

    exceptions = []

    # ======================================================
    # HANDLE PACKETS
    # ======================================================

    for packet in raw_packets:

        sales_order = (
            _validate_sales_order(
                packet.get(
                    "sales_order"
                )
            )
        )

        pages = sorted(
            set(
                packet.get(
                    "pages"
                )
                or []
            )
        )

        # --------------------------------------------------
        # NO SALES ORDER
        # --------------------------------------------------

        if not sales_order:

            review_pages.update(
                pages
            )

            if pages:

                exceptions.append({

                    "type":
                        "unassigned_pages",

                    "message":
                        (
                            "Pages could not be confidently assigned "
                            "to a Sales Order."
                        ),

                    "pages":
                        pages,
                })

            continue

        # --------------------------------------------------
        # SAME SO APPEARED IN MULTIPLE BLOCKS
        # --------------------------------------------------

        if sales_order in duplicate_so_set:

            review_pages.update(
                pages
            )

            continue

        # --------------------------------------------------
        # NORMAL VALID PACKET
        # --------------------------------------------------

        valid_packets.append({

            "sales_order":
                sales_order,

            "filename":
                f"{sales_order}.pdf",

            "pages":
                pages,

            "page_count":
                len(
                    pages
                ),
        })

    # ======================================================
    # ADD ONE DUPLICATE-SO EXCEPTION PER SALES ORDER
    # ======================================================

    for sales_order in duplicate_sos:

        ambiguous_pages = []

        for packet in raw_packets:

            if (
                _validate_sales_order(
                    packet.get(
                        "sales_order"
                    )
                )
                == sales_order
            ):

                ambiguous_pages.extend(
                    packet.get(
                        "pages"
                    )
                    or []
                )

        ambiguous_pages = sorted(
            set(
                ambiguous_pages
            )
        )

        review_pages.update(
            ambiguous_pages
        )

        exceptions.append({

            "type":
                "noncontiguous_duplicate_sales_order",

            "message":
                (
                    f"{sales_order} appeared in more than one "
                    "non-contiguous packet. Those pages require review."
                ),

            "sales_order":
                sales_order,

            "pages":
                ambiguous_pages,
        })

    # ======================================================
    # CHECK THAT EVERY PAGE IS REPRESENTED
    # ======================================================

    valid_packet_pages = set()

    for packet in valid_packets:

        valid_packet_pages.update(
            packet[
                "pages"
            ]
        )

    represented_pages = (
        valid_packet_pages
        | review_pages
    )

    expected_pages = set(
        range(
            1,
            total_pages + 1,
        )
    )

    missing_pages = sorted(
        expected_pages
        - represented_pages
    )

    if missing_pages:

        review_pages.update(
            missing_pages
        )

        exceptions.append({

            "type":
                "missing_pages",

            "message":
                (
                    "Pages were not represented in the initial "
                    "packet grouping and were moved to review."
                ),

            "pages":
                missing_pages,
        })

    # ======================================================
    # OVERLAP CHECK
    # ======================================================

    overlap_pages = sorted(
        valid_packet_pages
        & review_pages
    )

    if overlap_pages:

        # Safety rule:
        # review wins.
        #
        # Remove any affected entire packet from automated
        # output because its assignment is now ambiguous.

        surviving_packets = []

        for packet in valid_packets:

            packet_pages = set(
                packet[
                    "pages"
                ]
            )

            if packet_pages & set(
                overlap_pages
            ):

                review_pages.update(
                    packet_pages
                )

                exceptions.append({

                    "type":
                        "packet_overlap",

                    "message":
                        (
                            f"{packet['sales_order']} contained pages "
                            "that were also marked for review. "
                            "The entire packet was moved to review."
                        ),

                    "sales_order":
                        packet[
                            "sales_order"
                        ],

                    "pages":
                        packet[
                            "pages"
                        ],
                })

            else:

                surviving_packets.append(
                    packet
                )

        valid_packets = (
            surviving_packets
        )

    # ======================================================
    # SORT AUTOMATED OUTPUT
    # ======================================================

    valid_packets.sort(
        key=lambda packet:
            _sales_order_sort_key(
                packet[
                    "sales_order"
                ]
            )
    )

    review_pages = sorted(
        review_pages
    )

    # ======================================================
    # FINAL COVERAGE CHECK
    # ======================================================

    final_assigned_pages = set()

    for packet in valid_packets:

        final_assigned_pages.update(
            packet[
                "pages"
            ]
        )

    final_covered_pages = (
        final_assigned_pages
        | set(
            review_pages
        )
    )

    final_missing_pages = sorted(
        expected_pages
        - final_covered_pages
    )

    if final_missing_pages:

        review_pages = sorted(
            set(
                review_pages
            )
            | set(
                final_missing_pages
            )
        )

        exceptions.append({

            "type":
                "final_coverage_repair",

            "message":
                (
                    "Pages were missing from final coverage and "
                    "were automatically moved to review."
                ),

            "pages":
                final_missing_pages,
        })

    # ======================================================
    # STATUS
    # ======================================================

    if (
        valid_packets
        and not review_pages
        and not exceptions
    ):

        status = "SUCCESS"

    elif valid_packets:

        status = "PARTIAL_SUCCESS"

    else:

        status = "FAILED"

    # ======================================================
    # CUSTOMER-FACING SUMMARY
    # ======================================================

    customer_summary = {

        "status":
            status,

        "pages_received":
            total_pages,

        "sales_orders_created":
            len(
                valid_packets
            ),

        "pages_assigned":
            sum(
                packet[
                    "page_count"
                ]
                for packet in valid_packets
            ),

        "review_page_count":
            len(
                review_pages
            ),

        "review_pages":
            review_pages,

        "exception_count":
            len(
                exceptions
            ),
    }

    return {

        "ok":
            status
            in (
                "SUCCESS",
                "PARTIAL_SUCCESS",
            ),

        "status":
            status,

        "workflow":
            "ulp_packet_split",

        "page_count":
            total_pages,

        "packet_count":
            len(
                valid_packets
            ),

        "sales_orders":
            [
                packet[
                    "sales_order"
                ]
                for packet in valid_packets
            ],

        "assigned_page_count":
            customer_summary[
                "pages_assigned"
            ],

        "review_page_count":
            len(
                review_pages
            ),

        "review_pages":
            review_pages,

        "exception_count":
            len(
                exceptions
            ),

        "exceptions":
            exceptions,

        "packets":
            valid_packets,

        "customer_summary":
            customer_summary,

        "google_ocr":
            detection.get(
                "google_ocr"
            ),

        "page_diagnostics":
            page_results,
    }


# ==========================================================
# BUILD PDF FROM SELECTED PAGES
# ==========================================================

def _build_pdf_from_pages(
    reader: PdfReader,
    page_numbers: List[int],
) -> bytes:

    writer = PdfWriter()

    for page_number in page_numbers:

        writer.add_page(
            reader.pages[
                page_number
                - 1
            ]
        )

    output = io.BytesIO()

    writer.write(
        output
    )

    return output.getvalue()


# ==========================================================
# BUILD ZIP
# ==========================================================

def split_packet_to_zip(
    pdf_bytes: bytes,
) -> Dict[str, Any]:

    manifest = (
        analyze_packet(
            pdf_bytes
        )
    )

    # ======================================================
    # TOTAL FAILURE
    # ======================================================

    if manifest[
        "status"
    ] == "FAILED":

        return {

            "ok":
                False,

            "status":
                "FAILED",

            "manifest":
                manifest,

            "zip_bytes":
                None,
        }

    reader = PdfReader(
        io.BytesIO(
            pdf_bytes
        )
    )

    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(
        zip_buffer,
        mode="w",
        compression=
            zipfile.ZIP_DEFLATED,
    ) as zip_file:

        # ==================================================
        # VALID SALES ORDER PDFs
        # ==================================================

        for packet in manifest[
            "packets"
        ]:

            packet_pdf = (
                _build_pdf_from_pages(
                    reader,
                    packet[
                        "pages"
                    ],
                )
            )

            zip_file.writestr(
                packet[
                    "filename"
                ],
                packet_pdf,
            )

        # ==================================================
        # REVIEW PDF
        # ==================================================

        if manifest[
            "review_pages"
        ]:

            review_pdf = (
                _build_pdf_from_pages(
                    reader,
                    manifest[
                        "review_pages"
                    ],
                )
            )

            zip_file.writestr(
                "REVIEW_REQUIRED_PAGES.pdf",
                review_pdf,
            )

        # ==================================================
        # MANIFEST
        # ==================================================

        zip_file.writestr(
            "manifest.json",

            json.dumps(
                manifest,
                indent=2,
            ).encode(
                "utf-8"
            ),
        )

    return {

        "ok":
            True,

        "status":
            manifest[
                "status"
            ],

        "manifest":
            manifest,

        "zip_bytes":
            zip_buffer.getvalue(),
    }
