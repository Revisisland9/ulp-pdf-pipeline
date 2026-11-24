import json
import unittest
from email.message import EmailMessage
from types import SimpleNamespace
from unittest.mock import patch

from app.shipment_intake import (
    Address,
    Commodity,
    DocumentInputError,
    parse_document_to_shipment,
)


PDF = b"%PDF-1.7\n"
PNG = b"\x89PNG\r\n\x1a\n"
JPEG = b"\xff\xd8\xff"


def shipment_data():
    address = dict.fromkeys(Address.model_fields)
    item = dict.fromkeys(Commodity.model_fields)
    item.update(quantity=2, quantity_uom="PLT", weight=485.1, weight_uom="lb")
    return {
        "reference": {"load_id": "00123", "sales_order": None, "purchase_order": None},
        "shipper": {**address, "postal_code": "02108"},
        "consignee": {**address, "postal_code": "60490"},
        "source_services": ["Notify Consignee"],
        "pickup": {"earliest": None, "latest": None},
        "delivery": {"earliest": None, "latest": None},
        "items": [item, {**item, "quantity": 1, "weight": 90.2}],
        "expected_total_quantity": 3,
        "expected_total_quantity_uom": "pallets",
        "expected_total_weight": 575.3,
        "expected_total_weight_uom": "lbs",
        "warnings": [],
    }


class ShipmentIntakeTests(unittest.TestCase):
    def setUp(self):
        client = patch("app.shipment_intake._get_client")
        self.addCleanup(client.stop)
        self.create = client.start().return_value.responses.create
        self.respond(shipment_data())

    def respond(self, data):
        self.create.return_value = SimpleNamespace(
            status="completed", output_text=json.dumps(data)
        )

    def test_supported_direct_uploads(self):
        for filename, data, kind in [
            ("shipment.pdf", PDF, "input_file"),
            ("shipment.PNG", PNG, "input_image"),
            ("shipment.jpg", JPEG, "input_image"),
            ("shipment.jpeg", JPEG, "input_image"),
        ]:
            with self.subTest(filename=filename):
                shipment = parse_document_to_shipment(filename, data)
                self.assertEqual(shipment.status, "REVIEW_REQUIRED")
                self.assertEqual(shipment.shipper.postal_code, "02108")
                self.assertIsNone(shipment.items[0].hazardous_material)
                request = self.create.call_args.kwargs
                self.assertEqual(request["input"][0]["content"][0]["type"], kind)
                self.assertTrue(request["text"]["format"]["strict"])
                self.assertFalse(request["store"])

    def test_email_body_inline_image_and_attachment(self):
        message = EmailMessage()
        message["Subject"] = "Shipment 00123"
        message.set_content("Lift Gate Destination")
        message.add_related(PNG, maintype="image", subtype="png", cid="<shipment>")
        message.add_attachment(PDF, maintype="application", subtype="pdf", filename="shipment.pdf")
        message.add_attachment(b"spreadsheet", maintype="application", subtype="octet-stream", filename="shipment.xls")
        shipment = parse_document_to_shipment("shipment.eml", message.as_bytes())
        self.assertEqual(shipment.source.subject, "Shipment 00123")
        self.assertIn("shipment.xls", shipment.warnings[0])
        parts = self.create.call_args.kwargs["input"][0]["content"]
        self.assertEqual([part["type"] for part in parts], [
            "input_text", "input_text", "input_image", "input_file"
        ])

    def test_missing_fields_and_inconsistent_totals(self):
        data = shipment_data()
        data["shipper"]["postal_code"] = None
        data["items"][1]["weight_uom"] = "kg"
        data["expected_total_quantity"] = 4
        self.respond(data)
        shipment = parse_document_to_shipment(
            "shipment.pdf", PDF, require_freight_class=True, require_nmfc=True
        )
        self.assertEqual(shipment.status, "VALIDATION_FAILED")
        self.assertIn("Shipper postal code is required.", shipment.errors)
        self.assertIn("Commodity 1 freight class is required.", shipment.errors)
        self.assertIn("Commodity 1 NMFC is required.", shipment.errors)
        self.assertIn("Source quantity total 4 differs from commodity sum 3.0.", shipment.errors)
        self.assertIn(
            "Cannot reconcile source weight total across incompatible units.",
            shipment.errors,
        )

    def test_quantity_summary_label_is_not_a_unit(self):
        data = shipment_data()
        data["expected_total_quantity_uom"] = "qty"
        data["items"][1]["quantity_uom"] = "Pallets"
        self.respond(data)
        shipment = parse_document_to_shipment("shipment.pdf", PDF)
        self.assertEqual(shipment.errors, [])

        data["items"][1]["quantity_uom"] = None
        self.respond(data)
        shipment = parse_document_to_shipment("shipment.pdf", PDF)
        self.assertEqual(shipment.errors, [
            "Cannot reconcile source quantity total. Enter the quantity unit for commodity rows 2.",
        ])

    def test_dimension_unit_defaults_only_for_complete_dimensions(self):
        item = {**shipment_data()["items"][0], "length": 30, "width": 30, "height": 42}
        self.assertEqual(Commodity.model_validate(item).dimension_uom, "in")
        self.assertEqual(Commodity.model_validate({**item, "dimension_uom": "cm"}).dimension_uom, "cm")
        self.assertIsNone(Commodity.model_validate({**item, "height": None}).dimension_uom)

    def test_bad_upload_never_reaches_openai(self):
        with self.assertRaises(DocumentInputError):
            parse_document_to_shipment("shipment.pdf", PNG)
        with self.assertRaises(DocumentInputError):
            parse_document_to_shipment("shipment.eml", b"Subject: Empty\n\n")
        self.create.assert_not_called()

    def test_incomplete_extraction_cannot_return_a_review_result(self):
        self.create.return_value = SimpleNamespace(status="incomplete", output_text="{}")
        with self.assertRaisesRegex(RuntimeError, "completed shipment extraction"):
            parse_document_to_shipment("shipment.pdf", PDF)


if __name__ == "__main__":
    unittest.main()
