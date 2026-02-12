from io import BytesIO
from typing import Any, Dict, List

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable

from app.util.helpers import s, get_path, fmt_phone


def _primary_ref(req: Dict[str, Any]) -> str:
    for r in (req.get("ReferenceNumbers") or []):
        if r.get("IsPrimary"):
            return s(r.get("ReferenceNumber"))
    return ""


def _first_pickup_date(req: Dict[str, Any]) -> str:
    # Date: pull from "Early Pickup" (EarliestPickupDate)
    return s(get_path(req, "Dates", "EarliestPickupDate", default=""))


def _exclude_reference_type_pdf_(ref_type: str) -> bool:
    """
    PDF-only filter: never show Job name / Load number references on the PDF.
    Uses substring matching to catch common variants.
    """
    t = (ref_type or "").strip().lower()
    if not t:
        return False

    blocked = [
        "job name",
        "job",
        "load number",
        "load #",
        "load",
    ]
    return any(b in t for b in blocked)


def build_shipment_confirmation_pdf(req: Dict[str, Any]) -> bytes:
    """
    NOTE: Keeping function name to avoid changing imports/routes.
    Output formatted as BILL OF LADING with Third Party Prepaid terms + required notices/signature verbiage.
    """
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=LETTER,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
        title="Bill of Lading",
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Small", parent=styles["Normal"], fontSize=9, leading=11))
    styles.add(ParagraphStyle(name="H2", parent=styles["Heading2"], spaceBefore=8, spaceAfter=6))
    styles.add(ParagraphStyle(name="BolTitle", parent=styles["Title"], fontSize=20, leading=22, spaceAfter=2))
    styles.add(ParagraphStyle(name="BolHeader", parent=styles["Normal"], fontSize=10, leading=12))

    # Fine print + legal blocks
    styles.add(
        ParagraphStyle(
            name="FinePrint",
            parent=styles["Normal"],
            fontSize=7.5,
            leading=9,
            spaceBefore=2,
            spaceAfter=2,
        )
    )

    # NOTE bar style (white on black)
    styles.add(
        ParagraphStyle(
            name="NoteBar",
            parent=styles["Normal"],
            fontSize=9,
            leading=11,
            textColor=colors.white,
        )
    )

    story: List[Any] = []

    # ---------------- HEADER ----------------
    pref = _primary_ref(req)
    pickup_date = _first_pickup_date(req)

    story.append(Paragraph("BILL OF LADING", styles["BolTitle"]))

    left = f"<b>Primary Reference:</b> {pref}" if pref else "<b>Primary Reference:</b> —"
    date_str = pickup_date if pickup_date else "—"
    right = f"<b>Date:</b> {date_str}<br/><b>Terms:</b> Third Party Prepaid"

    header_tbl = Table(
        [[Paragraph(left, styles["BolHeader"]), Paragraph(right, styles["BolHeader"])]],
        colWidths=[3.6 * inch, 3.6 * inch],
    )
    header_tbl.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("PADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    story.append(header_tbl)

    story.append(Spacer(1, 0.08 * inch))
    story.append(HRFlowable(width="100%", thickness=1.1, color=colors.black))
    story.append(Spacer(1, 0.12 * inch))

    # ---------------- SHIPPER / CONSIGNEE / BILL TO ----------------
    shipper = req.get("Shipper") or {}
    consignee = req.get("Consignee") or {}
    bill_to = get_path(req, "Payment", "Address", default={}) or {}

    def party_block(p: Dict[str, Any], include_residential: bool) -> str:
        lines: List[str] = []
        if s(p.get("Name")):
            lines.append(s(p.get("Name")))
        if s(p.get("AddressLine1")):
            lines.append(s(p.get("AddressLine1")))
        if s(p.get("AddressLine2")):
            lines.append(s(p.get("AddressLine2")))

        city = s(p.get("City"))
        st = s(p.get("StateProvince"))
        pc = s(p.get("PostalCode"))
        cityline = " ".join([x for x in [f"{city},", st, pc] if x]).strip().replace(" ,", ",")
        if cityline:
            lines.append(cityline)

        if s(p.get("CountryCode")):
            lines.append(s(p.get("CountryCode")))

        ph = fmt_phone(get_path(p, "Contact", "Phone", default=""))
        if ph:
            lines.append(f"Phone: {ph}")

        if include_residential and p.get("IsResidential") is True:
            lines.append("Residential: Yes")

        return "<br/>".join(lines) if lines else ""

    parties_table = Table(
        [
            [
                Paragraph("<b>Shipper</b>", styles["Normal"]),
                Paragraph("<b>Consignee</b>", styles["Normal"]),
                Paragraph("<b>Bill To</b>", styles["Normal"]),  # changed from "THR Bill To"
            ],
            [
                Paragraph(party_block(shipper, include_residential=True), styles["Small"]),
                Paragraph(party_block(consignee, include_residential=True), styles["Small"]),
                Paragraph(party_block(bill_to, include_residential=False), styles["Small"]),
            ],
        ],
        colWidths=[2.4 * inch, 2.4 * inch, 2.4 * inch],
    )
    parties_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                ("PADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(parties_table)

    # ---------------- REFERENCES (no black bar, no Primary column) on RIGHT HALF under parties ----------------
    refs_all = req.get("ReferenceNumbers") or []
    # PDF-only: remove Job name / Load number
    refs = [r for r in refs_all if not _exclude_reference_type_pdf_(s(r.get("Type")))]

    if refs:
        story.append(Spacer(1, 0.12 * inch))

        ref_rows = [["Type", "Reference Number"]]
        for r in refs:
            ref_rows.append([s(r.get("Type")), s(r.get("ReferenceNumber"))])

        ref_table = Table(ref_rows, colWidths=[1.6 * inch, 2.0 * inch])
        ref_table.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("PADDING", (0, 0), (-1, -1), 6),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                ]
            )
        )

        right_half = Table([[Paragraph("", styles["Small"]), ref_table]], colWidths=[3.6 * inch, 3.6 * inch])
        right_half.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("PADDING", (0, 0), (-1, -1), 0),
                ]
            )
        )
        story.append(right_half)

    # ---------------- ITEMS (keep as-is) ----------------
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("Items", styles["H2"]))

    items = req.get("Items") or []
    if not items:
        story.append(Paragraph("No items provided.", styles["Small"]))
    else:
        rows = [["Description", "Qty", "Wt (lb)", "Dims (in)", "Class", "NMFC"]]
        for it in items:
            desc = s(it.get("Description"))
            qty = s(get_path(it, "Quantities", "Actual", default=""))
            qty_u = s(get_path(it, "Quantities", "Uom", default=""))
            wt = s(get_path(it, "Weights", "Actual", default=""))
            dims = it.get("Dimensions") or {}
            dims_str = f'{s(dims.get("Length"))}x{s(dims.get("Width"))}x{s(dims.get("Height"))}'
            fc = s(get_path(it, "FreightClasses", "FreightClass", default=""))
            nmfc = s(it.get("NmfcCode"))

            rows.append([desc, f"{qty} {qty_u}".strip(), wt, dims_str, fc, nmfc])

        itab = Table(rows, colWidths=[2.9 * inch, 0.9 * inch, 0.8 * inch, 1.0 * inch, 0.6 * inch, 1.1 * inch])
        itab.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.black),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 9),
                    ("FONTSIZE", (0, 1), (-1, -1), 9),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
                    ("PADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        story.append(itab)

    # ---------------- NOTE BAR + FINE PRINT BLOCK (under items) ----------------
    story.append(Spacer(1, 0.12 * inch))

    note_text = (
        "NOTE: Liability limitation for loss or damage in this shipment may be applicable. "
        "See 49 USC 14706(c)(1)(A) and (B)."
    )
    note_tbl = Table([[Paragraph(note_text, styles["NoteBar"])]], colWidths=[7.2 * inch])
    note_tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.black),
                ("PADDING", (0, 0), (-1, -1), 6),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.append(note_tbl)

    fine_block = (
        "Received, subject to the agreement between the Carrier and listed Third Party. In effect on the date of shipment "
        "Carrier agrees that listed Third Party is the sole payer of the corresponding freight bill. "
        "This Bill of Lading is not subject to any tariffs or classifications, whether individually determined of filed with any federal "
        "or state regulatory agency, except as specifically agreed to in writing by the listed Third Party and Carrier."
    )
    story.append(Spacer(1, 0.06 * inch))
    story.append(Paragraph(fine_block, styles["FinePrint"]))

    # ---------------- SIGNATURES + REQUIRED VERBIAGE ----------------
    story.append(Spacer(1, 0.16 * inch))

    shipper_cert = (
        "This is to certify that the above named materials are properly classified, described, packaged, marked and labeled, "
        "and are in proper condition for transportation according to the applicable regulations of the Department of "
        "Transportation."
    )
    story.append(Paragraph(shipper_cert, styles["FinePrint"]))
    story.append(Spacer(1, 0.06 * inch))

    sig_shipper = Table(
        [["Shipper Signature: ________________________________", "Date: ________________"]],
        colWidths=[5.3 * inch, 1.9 * inch],
    )
    sig_shipper.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "BOTTOM"),
                ("PADDING", (0, 0), (-1, -1), 6),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
            ]
        )
    )
    story.append(sig_shipper)

    story.append(Spacer(1, 0.12 * inch))

    carrier_ack = (
        "Carrier acknowledges receipt of packages and required four (4) placards. Carrier certifies emergency response "
        "information was made available and/or carrier has the Department of Transportaton emergency response guidebook or "
        "equivalent documentation in vehicle. Property described above is received in good order, except as noted"
    )
    story.append(Paragraph(carrier_ack, styles["FinePrint"]))
    story.append(Spacer(1, 0.06 * inch))

    sig_consignee = Table(
        [["Consignee Signature: _____________________________", "Date: ________________"]],
        colWidths=[5.3 * inch, 1.9 * inch],
    )
    sig_consignee.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "BOTTOM"),
                ("PADDING", (0, 0), (-1, -1), 6),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
            ]
        )
    )
    story.append(sig_consignee)

    story.append(Spacer(1, 0.10 * inch))

    sig_driver = Table(
        [["Driver Signature:  ________________________________", "Date: ________________"]],
        colWidths=[5.3 * inch, 1.9 * inch],
    )
    sig_driver.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "BOTTOM"),
                ("PADDING", (0, 0), (-1, -1), 6),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
            ]
        )
    )
    story.append(sig_driver)

    doc.build(story)
    buf.seek(0)
    return buf.read()
