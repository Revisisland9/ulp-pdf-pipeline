from io import BytesIO
from typing import Any, Dict, List

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)

from app.util.helpers import s, get_path, fmt_phone

def build_shipment_confirmation_pdf(req: Dict[str, Any]) -> bytes:
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=LETTER,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
        title="Shipment Confirmation",
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Small", parent=styles["Normal"], fontSize=9, leading=11))
    styles.add(ParagraphStyle(name="H2", parent=styles["Heading2"], spaceBefore=10, spaceAfter=6))

    story: List[Any] = []

    # ---- Header ----
    story.append(Paragraph("Shipment Confirmation", styles["Title"]))

    status = s(req.get("Status"))
    if status:
        story.append(Paragraph(f"<b>Status:</b> {status}", styles["Normal"]))

    # Best “document id” = primary reference if present
    primary_ref = ""
    for r in (req.get("ReferenceNumbers") or []):
        if r.get("IsPrimary"):
            primary_ref = s(r.get("ReferenceNumber"))
            break
    if primary_ref:
        story.append(Paragraph(f"<b>Primary Reference:</b> {primary_ref}", styles["Normal"]))

    story.append(Spacer(1, 0.08 * inch))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.black))
    story.append(Spacer(1, 0.10 * inch))

    # ---- Reference Numbers ----
    refs = req.get("ReferenceNumbers") or []
    story.append(Paragraph("Reference Numbers", styles["H2"]))
    if refs:
        ref_rows = [["Type", "ReferenceNumber", "Primary"]]
        for r in refs:
            ref_rows.append([
                s(r.get("Type")),
                s(r.get("ReferenceNumber")),
                "Yes" if r.get("IsPrimary") else "",
            ])
        t = Table(ref_rows, colWidths=[1.7*inch, 3.9*inch, 0.8*inch])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.black),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,0), 9),
            ("FONTSIZE", (0,1), (-1,-1), 9),
            ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.whitesmoke, colors.white]),
            ("PADDING", (0,0), (-1,-1), 6),
        ]))
        story.append(t)
    else:
        story.append(Paragraph("No reference numbers provided.", styles["Small"]))

    # ---- Parties ----
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("Parties", styles["H2"]))

    shipper = req.get("Shipper") or {}
    consignee = req.get("Consignee") or {}
    payment_addr = get_path(req, "Payment", "Address", default={}) or {}

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
                Paragraph("<b>Bill To / Payment</b>", styles["Normal"]),
            ],
            [
                Paragraph(party_block(shipper, include_residential=True), styles["Small"]),
                Paragraph(party_block(consignee, include_residential=True), styles["Small"]),
                Paragraph(party_block(payment_addr, include_residential=False), styles["Small"]),
            ]
        ],
        colWidths=[2.4*inch, 2.4*inch, 2.4*inch]
    )
    parties_table.setStyle(TableStyle([
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("BOX", (0,0), (-1,-1), 0.5, colors.grey),
        ("INNERGRID", (0,0), (-1,-1), 0.25, colors.lightgrey),
        ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
        ("PADDING", (0,0), (-1,-1), 8),
    ]))
    story.append(parties_table)

    # ---- Schedule & Service ----
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("Schedule & Service", styles["H2"]))

    dates = req.get("Dates") or {}
    flags = req.get("ServiceFlags") or []
    selected_flags = ", ".join([s(f.get("ServiceCode")) for f in flags if f.get("IsSelected")]) or ""

    schedule_rows = [
        ["Earliest Pickup", s(dates.get("EarliestPickupDate"))],
        ["Latest Pickup",   s(dates.get("LatestPickupDate"))],
        ["Earliest Drop",   s(dates.get("EarliestDropDate"))],
        ["Latest Drop",     s(dates.get("LatestDropDate"))],
        ["Selected Service Flags", selected_flags],
    ]
    sched = Table(schedule_rows, colWidths=[1.9*inch, 5.3*inch])
    sched.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.25, colors.lightgrey),
        ("BACKGROUND", (0,0), (0,-1), colors.whitesmoke),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("PADDING", (0,0), (-1,-1), 6),
        ("FONTSIZE", (0,0), (-1,-1), 9),
    ]))
    story.append(sched)

    # ---- Items ----
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
            ("BACKGROUND", (0,0), (-1,0), colors.black),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,0), 9),
            ("FONTSIZE", (0,1), (-1,-1), 9),
            ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.whitesmoke, colors.white]),
            ("PADDING", (0,0), (-1,-1), 6),
        ]))
        story.append(itab)

    # ---- Meta ----
    meta = req.get("Meta") or {}
    if meta:
        story.append(Spacer(1, 0.18 * inch))
        story.append(Paragraph("Meta", styles["H2"]))
        mrows = [[s(k), s(v)] for k, v in meta.items()]
        mt = Table(mrows, colWidths=[1.9*inch, 5.3*inch])
        mt.setStyle(TableStyle([
            ("GRID", (0,0), (-1,-1), 0.25, colors.lightgrey),
            ("BACKGROUND", (0,0), (0,-1), colors.whitesmoke),
            ("PADDING", (0,0), (-1,-1), 6),
            ("FONTSIZE", (0,0), (-1,-1), 9),
        ]))
        story.append(mt)

    doc.build(story)
    buf.seek(0)
    return buf.read()
