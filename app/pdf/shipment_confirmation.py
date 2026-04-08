from io import BytesIO
from typing import Any, Dict, List, Optional

from reportlab.graphics.barcode import code128
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


# ---------------- Helpers ----------------

def _primary_ref(req: Dict[str, Any]) -> str:
    for r in (req.get("ReferenceNumbers") or []):
        if r.get("IsPrimary"):
            return s(r.get("ReferenceNumber"))
    return ""


def _first_pickup_date(req: Dict[str, Any]) -> str:
    return s(get_path(req, "Dates", "EarliestPickupDate", default=""))


def _date_only(dt: str) -> str:
    dt = (dt or "").strip()
    return dt.split()[0] if dt else ""


def _find_ref_value_(req: Dict[str, Any], type_names: List[str]) -> str:
    """
    Find a ReferenceNumber by matching Type (case-insensitive) against any provided names.
    """
    want = {s(n).strip().lower() for n in (type_names or []) if s(n).strip()}
    for r in (req.get("ReferenceNumbers") or []):
        t = s(r.get("Type")).strip().lower()
        if t in want:
            return s(r.get("ReferenceNumber"))
    return ""


def _services_display(req: Dict[str, Any]) -> str:
    """
    Always show 'Appointment Required' on all BOLs.
    If LG1 is present + selected anywhere we look, add 'Liftgate Required'.

    Supports ServiceFlags at:
      - req["ServiceFlags"]   (your current JSON)
      - req["Constraints"]["ServiceFlags"] (older shape)
    """
    services: List[str] = ["Appointment Required"]  # always

    flags = (
        (req.get("ServiceFlags") or [])
        or (get_path(req, "Constraints", "ServiceFlags", default=[]) or [])
    )

    liftgate_on = False
    for f in flags:
        if f.get("IsSelected") is not True:
            continue
        code = s(f.get("ServiceCode")).upper()
        if code == "LG1":
            liftgate_on = True
            break

    if liftgate_on:
        services.append("Liftgate Required")

    return ", ".join(services)


def _build_pro_barcode_block_(req: Dict[str, Any], styles) -> Optional[Any]:
    """
    Build a barcode flowable for PRO Number (Code128) that sits in the left empty space
    next to the references table.

    Changes requested:
      - No "PRO BARCODE" label
      - Center the PRO number text under the barcode
    """
    pro = _find_ref_value_(req, ["pro number", "pro", "pro#"])
    if not pro:
        return None

    # Barcode value should not include spaces
    pro_code = (pro or "").replace(" ", "").strip()
    if not pro_code:
        return None

    # Use a thinner barWidth so it fits reliably in the left column
    bc = code128.Code128(pro_code, barHeight=0.55 * inch, barWidth=0.6)

    blk = Table(
        [
            [bc],
            [Paragraph(pro, styles["SmallCenter"])],
        ],
        colWidths=[3.4 * inch],
    )
    blk.setStyle(TableStyle([
        ("PADDING", (0, 0), (-1, -1), 0),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))
    return blk


def _is_job_or_load_(ref_type: str) -> bool:
    """
    Hide these refs from the main references table because they render
    separately under the items table.

    Supports both old and new names:
      - old: Job Name / Load Number
      - new: Quantity / Location
    """
    t = (ref_type or "").strip().lower()
    return (
        ("job name" in t)
        or ("load number" in t)
        or ("quantity" in t)
        or ("location" in t)
    )


def _sum_item_weights_(req: Dict[str, Any]) -> float:
    """
    Fallback total weight calculation from Items.
    Supports your current item shape:
      - Weights.Actual
      - Quantities.Actual (defaults to 1)
    """
    total = 0.0
    for it in (req.get("Items") or []):
        wt = get_path(it, "Weights", "Actual", default=0)
        qty = get_path(it, "Quantities", "Actual", default=1)

        try:
            wt_num = float(wt or 0)
        except Exception:
            wt_num = 0.0

        try:
            qty_num = float(qty or 1)
        except Exception:
            qty_num = 1.0

        total += wt_num * qty_num

    return total


def _total_weight_display_(req: Dict[str, Any]) -> str:
    """
    Prefer Meta.TotalWeight from Apps Script.
    Fall back to summing Items if not present.
    """
    meta_total = get_path(req, "Meta", "TotalWeight", default=None)

    if meta_total not in (None, "", "null"):
        try:
            val = float(meta_total)
            if val.is_integer():
                return f"{int(val):,} lb"
            return f"{val:,.2f} lb"
        except Exception:
            pass

    fallback = _sum_item_weights_(req)
    if fallback.is_integer():
        return f"{int(fallback):,} lb"
    return f"{fallback:,.2f} lb"


# ---------------- PDF Builder ----------------

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
    styles.add(ParagraphStyle(name="SmallCenter", parent=styles["Normal"], fontSize=9, leading=11, alignment=1))
    styles.add(ParagraphStyle(name="BolTitle", parent=styles["Title"], fontSize=20, leading=22))
    styles.add(ParagraphStyle(name="BolHeader", parent=styles["Normal"], fontSize=10, leading=12))
    styles.add(ParagraphStyle(name="FinePrint", parent=styles["Normal"], fontSize=7.5, leading=9))

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
    pickup_date = _date_only(_first_pickup_date(req))

    header_tbl = Table(
        [[
            Paragraph(f"<b>Primary Reference:</b> {pref or '—'}", styles["BolHeader"]),
            Paragraph(f"<b>Date:</b> {pickup_date or '—'}", styles["BolHeader"]),
            Paragraph("<b>Terms:</b> Third Party Prepaid", styles["BolHeader"]),
        ]],
        colWidths=[2.8 * inch, 2.2 * inch, 2.2 * inch],
    )
    header_tbl.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 0),
    ]))
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

        ph = fmt_phone(get_path(p, "Contact", "Phone", default=""))
        if ph:
            lines.append(f"Phone: {ph}")

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
        ("BACKGROUND", (0, 0), (-1, 0), colors.black),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ("PADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(parties_table)

    # Grab Quantity/Location values for later (under the items table)
    # Supports both new and old reference names during transition.
    qty_val = _find_ref_value_(req, ["quantity", "job name"])         # prints as QTY
    plt_loc_val = _find_ref_value_(req, ["location", "load number"])  # prints as PLT LOC.

    # ---------------- PRO BARCODE (LEFT) + REFERENCES + SERVICES (RIGHT) ----------------
    refs_all = req.get("ReferenceNumbers") or []
    refs = [r for r in refs_all if not _is_job_or_load_(s(r.get("Type")))]

    left_block = _build_pro_barcode_block_(req, styles) or Paragraph("", styles["Small"])

    right_stack: List[Any] = []

    if refs:
        ref_rows = []
        for r in refs:
            t_raw = s(r.get("Type")).strip()
            v = s(r.get("ReferenceNumber"))
            if not t_raw and not v:
                continue
            ref_rows.append([
                Paragraph(f"<b>{t_raw}:</b>", styles["Small"]),
                Paragraph(v or "—", styles["Small"]),
            ])

        if ref_rows:
            ref_table = Table(ref_rows, colWidths=[1.8 * inch, 1.8 * inch])
            ref_table.setStyle(TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                ("PADDING", (0, 0), (-1, -1), 6),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]))
            right_stack.append(ref_table)

    right_stack.append(Spacer(1, 0.08 * inch))
    right_stack.append(Paragraph(f"<b>Services:</b> {_services_display(req)}", styles["BolHeader"]))

    story.append(Spacer(1, 0.12 * inch))

    two_col = Table(
        [[left_block, right_stack]],
        colWidths=[3.6 * inch, 3.6 * inch],
    )
    two_col.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(two_col)

    # ---------------- ITEMS TABLE (NO 'Items' WORD) ----------------
    story.append(Spacer(1, 0.20 * inch))

    items = req.get("Items") or []
    rows = [["Description", "Qty", "Wt (lb)", "Dims (in)", "Class", "NMFC"]]
    for it in items:
        rows.append([
            s(it.get("Description")),
            f"{s(get_path(it, 'Quantities', 'Actual', default=''))} {s(get_path(it, 'Quantities', 'Uom', default=''))}".strip(),
            s(get_path(it, "Weights", "Actual", default="")),
            f"{s(get_path(it, 'Dimensions', 'Length', default=''))}x{s(get_path(it, 'Dimensions', 'Width', default=''))}x{s(get_path(it, 'Dimensions', 'Height', default=''))}",
            s(get_path(it, "FreightClasses", "FreightClass", default="")),
            s(it.get("NmfcCode")),
        ])

    itab = Table(
        rows,
        colWidths=[2.9 * inch, 0.9 * inch, 0.8 * inch, 1.0 * inch, 0.6 * inch, 1.1 * inch],
    )
    itab.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.black),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(itab)

    # ---------------- QTY + PLT LOC ROW ----------------
    story.append(Spacer(1, 0.10 * inch))
    qty_disp = qty_val or "—"
    plt_disp = plt_loc_val or "—"

    qty_pltl_tbl = Table(
        [[
            Paragraph(f"<b>QTY:</b> {qty_disp}", styles["BolHeader"]),
            Paragraph(f"<b>PLT LOC.:</b> {plt_disp}", styles["BolHeader"]),
        ]],
        colWidths=[3.6 * inch, 3.6 * inch],
    )
    qty_pltl_tbl.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(qty_pltl_tbl)

    # ---------------- TOTAL WEIGHT ROW ----------------
    story.append(Spacer(1, 0.06 * inch))
    total_weight_tbl = Table(
        [[
            Paragraph(f"<b>Total Weight:</b> {_total_weight_display_(req)}", styles["BolHeader"]),
            Paragraph("", styles["BolHeader"]),
        ]],
        colWidths=[3.6 * inch, 3.6 * inch],
    )
    total_weight_tbl.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(total_weight_tbl)

    # ---------------- NOTE BAR + LEGAL TEXT ----------------
    story.append(Spacer(1, 0.30 * inch))
    note_tbl = Table([[
        Paragraph(
            "NOTE: Liability limitation for loss or damage in this shipment may be applicable. "
            "See 49 USC 14706(c)(1)(A) and (B).",
            styles["NoteBar"],
        )
    ]], colWidths=[7.2 * inch])
    note_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.black),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(note_tbl)

    story.append(Spacer(1, 0.10 * inch))
    story.append(Paragraph(
        "Received, subject to the agreement between the Carrier and listed Third Party. "
        "In effect on the date of shipment Carrier agrees that listed Third Party is the sole payer "
        "of the corresponding freight bill. This Bill of Lading is not subject to any tariffs or classifications, "
        "whether individually determined or filed with any federal or state regulatory agency, except as "
        "specifically agreed to in writing by the listed Third Party and Carrier.",
        styles["FinePrint"],
    ))

    # Space before signature blocks
    story.append(Spacer(1, 0.40 * inch))

    # ---------------- SHIPPER SIGNATURE BOX ----------------
    shipper_box = Table(
        [[
            Paragraph(
                "This is to certify that the above named materials are properly classified, described, packaged, "
                "marked and labeled, and are in proper condition for transportation according to the applicable "
                "regulations of the Department of Transportation.",
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

    # ---------------- DRIVER SIGNATURE BOX ----------------
    driver_box = Table(
        [[
            Paragraph(
                "Carrier acknowledges receipt of packages and required four (4) placards. Carrier certifies emergency "
                "response information was made available and/or carrier has the Department of Transportation emergency "
                "response guidebook or equivalent documentation in vehicle. Property described above is received in good "
                "order, except as noted.",
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
