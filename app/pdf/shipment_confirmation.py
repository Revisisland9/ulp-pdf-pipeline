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
    return s(get_path(req, "Dates", "EarliestPickupDate", default=""))


def _exclude_reference_type_pdf_(ref_type: str) -> bool:
    t = (ref_type or "").strip().lower()
    return ("job name" in t) or ("load number" in t)


def _services_display(req: Dict[str, Any]) -> str:
    labels: List[str] = ["Appointment Required"]
    flags = get_path(req, "Constraints", "ServiceFlags", default=[]) or []

    for f in flags:
        code = s(f.get("ServiceCode")).upper()
        if f.get("IsSelected") is True and code in {"LIFTGATE", "LIFT", "LG"}:
            if "Liftgate" not in labels:
                labels.append("Liftgate")

    return ", ".join(labels)


def build_shipment_confirmation_pdf(req: Dict[str, Any]) -> bytes:
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
    styles.add(ParagraphStyle(name="BolTitle", parent=styles["Title"], fontSize=20, leading=22))
    styles.add(ParagraphStyle(name="BolHeader", parent=styles["Normal"], fontSize=10, leading=12))

    styles.add(ParagraphStyle(
        name="FinePrint",
        parent=styles["Normal"],
        fontSize=7.5,
        leading=9,
    ))

    styles.add(ParagraphStyle(
        name="NoteBar",
        parent=styles["Normal"],
        fontSize=9,
        leading=11,
        textColor=colors.white,
        fontName="Helvetica-Bold",
    ))

    story: List[Any] = []

    # ---------------- TITLE ----------------
    story.append(Paragraph("BILL OF LADING", styles["BolTitle"]))
    story.append(Spacer(1, 0.30 * inch))

    # ---------------- HEADER ROW ----------------
    pref = _primary_ref(req)
    pickup_date = _first_pickup_date(req)

    header_tbl = Table(
        [[
            Paragraph(f"<b>Primary Reference:</b> {pref or '—'}", styles["BolHeader"]),
            Paragraph(f"<b>Date:</b> {pickup_date or '—'}", styles["BolHeader"]),
            Paragraph("<b>Terms:</b> Third Party Prepaid", styles["BolHeader"]),
        ]],
        colWidths=[2.8 * inch, 2.2 * inch, 2.2 * inch],
    )
    story.append(header_tbl)
    story.append(Spacer(1, 0.08 * inch))
    story.append(HRFlowable(width="100%", thickness=1.1, color=colors.black))
    story.append(Spacer(1, 0.12 * inch))

    # ---------------- PARTIES ----------------
    shipper = req.get("Shipper") or {}
    consignee = req.get("Consignee") or {}
    bill_to = get_path(req, "Payment", "Address", default={}) or {}

    def party_block(p: Dict[str, Any], include_residential: bool) -> str:
        lines = []
        if s(p.get("Name")): lines.append(s(p.get("Name")))
        if s(p.get("AddressLine1")): lines.append(s(p.get("AddressLine1")))
        if s(p.get("AddressLine2")): lines.append(s(p.get("AddressLine2")))

        city = s(p.get("City"))
        st = s(p.get("StateProvince"))
        pc = s(p.get("PostalCode"))
        cityline = " ".join([x for x in [f"{city},", st, pc] if x]).strip().replace(" ,", ",")
        if cityline: lines.append(cityline)

        ph = fmt_phone(get_path(p, "Contact", "Phone", default=""))
        if ph: lines.append(f"Phone: {ph}")

        if include_residential and p.get("IsResidential") is True:
            lines.append("Residential: Yes")

        return "<br/>".join(lines) if lines else ""

    parties_table = Table(
        [
            ["Shipper", "Consignee", "Bill To"],
            [
                Paragraph(party_block(shipper, True), styles["Small"]),
                Paragraph(party_block(consignee, True), styles["Small"]),
                Paragraph(party_block(bill_to, False), styles["Small"]),
            ],
        ],
        colWidths=[2.4 * inch, 2.4 * inch, 2.4 * inch],
    )
    parties_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(parties_table)

    # ---------------- ITEMS ----------------
    story.append(Spacer(1, 0.20 * inch))
    story.append(Paragraph("Items", styles["H2"]))

    rows = [["Description", "Qty", "Wt (lb)", "Dims (in)", "Class", "NMFC"]]
    for it in (req.get("Items") or []):
        rows.append([
            s(it.get("Description")),
            f"{s(get_path(it,'Quantities','Actual',default=''))} {s(get_path(it,'Quantities','Uom',default=''))}".strip(),
            s(get_path(it, "Weights", "Actual", default="")),
            f"{s(get_path(it,'Dimensions','Length',default=''))}x{s(get_path(it,'Dimensions','Width',default=''))}x{s(get_path(it,'Dimensions','Height',default=''))}",
            s(get_path(it, "FreightClasses", "FreightClass", default="")),
            s(it.get("NmfcCode")),
        ])

    itab = Table(rows,
                 colWidths=[2.9 * inch, 0.9 * inch, 0.8 * inch,
                            1.0 * inch, 0.6 * inch, 1.1 * inch])
    itab.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.black),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(itab)

    story.append(Spacer(1, 0.40 * inch))

    # ---------------- SHIPPER BOX ----------------
    shipper_box = Table(
        [[
            Paragraph(
                "This is to certify that the above named materials are properly classified, described, "
                "packaged, marked and labeled, and are in proper condition for transportation according "
                "to the applicable regulations of the Department of Transportation.",
                styles["FinePrint"],
            )
        ],
         [Spacer(1, 0.15 * inch)],
         [Table(
             [["Shipper Signature: ________________________________", "Date: ________________"]],
             colWidths=[5.3 * inch, 1.9 * inch],
         )]],
        colWidths=[7.2 * inch],
    )

    shipper_box.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 1, colors.black),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))

    story.append(shipper_box)
    story.append(Spacer(1, 0.25 * inch))

    # ---------------- DRIVER BOX ----------------
    driver_box = Table(
        [[
            Paragraph(
                "Carrier acknowledges receipt of packages and required four (4) placards. Carrier certifies "
                "emergency response information was made available and/or carrier has the Department of "
                "Transportation emergency response guidebook or equivalent documentation in vehicle. "
                "Property described above is received in good order, except as noted.",
                styles["FinePrint"],
            )
        ],
         [Spacer(1, 0.15 * inch)],
         [Table(
             [["Driver Signature: ________________________________", "Date: ________________"]],
             colWidths=[5.3 * inch, 1.9 * inch],
         )]],
        colWidths=[7.2 * inch],
    )

    driver_box.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 1, colors.black),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))

    story.append(driver_box)

    doc.build(story)
    buf.seek(0)
    return buf.read()
