from io import BytesIO
from typing import Any, Dict, List

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
)

from app.util.helpers import s, get_path, fmt_phone


def _primary_ref(req: Dict[str, Any]) -> str:
    for r in (req.get("ReferenceNumbers") or []):
        if r.get("IsPrimary"):
            return s(r.get("ReferenceNumber"))
    return ""


def _first_pickup_date(req: Dict[str, Any]) -> str:
    # You asked: Date: pull from "Early Pickup" (EarliestPickupDate)
    return s(get_path(req, "Dates", "EarliestPickupDate", default=""))


def build_shipment_confirmation_pdf(req: Dict[str, Any]) -> bytes:
    """
    NOTE: Keeping function name to avoid changing imports/routes.
    Output now formatted as BILL OF LADING per your requested layout.
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

    story: List[Any] = []

    # ---------------- HEADER ----------------
    pref = _primary_ref(req)
    pickup_date = _first_pickup_date(req)

    story.append(Paragraph("BILL OF LADING", styles["BolTitle"]))

    # Top line: Primary Reference + Date (no Status)
    left = f"<b>Primary Reference:</b> {pref}" if pref else "<b>Primary Reference:</b> —"
    right = f"<b>Date:</b> {pickup_date}" if pickup_date else "<b>Date:</b> —"

    header_tbl = Table(
        [[Paragraph(left, styles["BolHeader"]), Paragraph(right, styles["BolHeader"])]],
        colWidths=[3.6 * inch, 3.6 * inch],
    )
    header_tbl.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(header_tbl)

    story.append(Spacer(1, 0.08 * inch))
    story.append(HRFlowable(width="100%", thickness=1.1, color=colors.black))
    story.append(Spacer(1, 0.12 * inch))

    # ---------------- SHIPPER / CONSIGNEE / THR BILL TO (moved up, no "Parties" header) ----------------
    shipper = req.get("Shipper") or {}
    consignee = req.get("Consignee") or {}
    thr_bill_to = get_path(req, "Payment", "Address", default={}) or {}

    def party_block(p: Dict[str, Any], include_residential: bool) -> str:
        lines = []
        if s(p.get("Name")): lines.append(s(p.get("Name")))
        if s(p.get("AddressLine1")): lines.append(s(p.get("AddressLine1")))
        if s(p.get("AddressLine2")): lines.append(s(p.get("AddressLine2")))
        city = s(p.get("City")); st = s(p.get("StateProvince")); pc = s(p.get("PostalCode"))
        cityline = " ".join([x for x in [f"{city},", st, pc] if x]).strip().replace(" ,", ",")
        if cityline: lines.append(cityline)
        if s(p.get("CountryCode")): lines.append(s(p.get("CountryCode")))

        ph = fmt_phone(get_path(p, "Contact", "Phone", default=""))
        if ph: lines.append(f"Phone: {ph}")

        if include_residential and p.get("IsResidential") is True:
            lines.append("Residential: Yes")

        return "<br/>".join(lines) if lines else ""

    parties_table = Table(
        [
            [
                Paragraph("<b>Shipper</b>", styles["Normal"]),
                Paragraph("<b>Consignee</b>", styles["Normal"]),
                Paragraph("<b>THR Bill To</b>", styles["Normal"]),
            ],
            [
                Paragraph(party_block(shipper, include_residential=True), styles["Small"]),
                Paragraph(party_block(consignee, include_residential=True), styles["Small"]),
                Paragraph(party_block(thr_bill_to, include_residential=False), styles["Small"]),
            ],
        ],
        colWidths=[2.4 * inch, 2.4 * inch, 2.4 * inch],
    )
    parties_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(parties_table)

    # ---------------- REFERENCES (no black bar, no Primary column) on RIGHT HALF under parties ----------------
    refs = req.get("ReferenceNumbers") or []
    if refs:
        story.append(Spacer(1, 0.12 * inch))

        ref_rows = [["Type", "Reference Number"]]
        for r in refs:
            ref_rows.append([s(r.get("Type")), s(r.get("ReferenceNumber"))])

        ref_table = Table(ref_rows, colWidths=[1.6 * inch, 2.0 * inch])
        ref_table.setStyle(TableStyle([
            # simple, not blacked-out
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("PADDING", (0, 0), (-1, -1), 6),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
        ]))

        # Place in right half: left cell blank, right cell contains the table
        right_half = Table(
            [[
                Paragraph("", styles["Small"]),
                ref_table
            ]],
            colWidths=[3.6 * inch, 3.6 * inch],
        )
        right_half.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("PADDING", (0, 0), (-1, -1), 0),
        ]))
        story.append(right_half)

    # ---------------- (Removed) Delivery dates / early/late pickup/drop sections ----------------

    # ---------------- ITEMS (keep as-is, per your note) ----------------
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

            rows.append([
                desc,
                f"{qty} {qty_u}".strip(),
                wt,
                dims_str,
                fc,
                nmfc,
            ])

        itab = Table(rows, colWidths=[2.9*inch, 0.9*inch, 0.8*inch, 1.0*inch, 0.6*inch, 1.1*inch])
        itab.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.black),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("FONTSIZE", (0, 1), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(itab)

    # ---------------- (Removed) Meta section ----------------

    # ---------------- SIGNATURES (bottom) ----------------
    story.append(Spacer(1, 0.35 * inch))

    sig_rows = [
        ["Shipper Signature: ________________________________", "Date: ________________"],
        ["Driver Signature:  ________________________________", "Date: ________________"],
    ]
    sig = Table(sig_rows, colWidths=[5.3 * inch, 1.9 * inch])
    sig.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "BOTTOM"),
        ("PADDING", (0, 0), (-1, -1), 6),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
    ]))
    story.append(sig)

    doc.build(story)
    buf.seek(0)
    return buf.read()
