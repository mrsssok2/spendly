"""Receipt scanning (vision.py + /expenses/scan). The real vision model is never called:
a scripted fake replies instead. Test images are generated with Pillow."""
import io
import json
from datetime import date, timedelta

import cv2
import httpx
import numpy as np
import pytest
from langchain_core.messages import AIMessage
from PIL import Image, ImageDraw

import database.db as db
import vision
from conftest import FakeToolModel, count_expenses

TODAY = date.today()


def make_image(fmt="PNG", size=(400, 600), text="CAFE MOCHA  NET AMOUNT 858.00"):
    img = Image.new("RGB", size, (245, 240, 228))
    ImageDraw.Draw(img).text((20, 40), text, fill=(40, 40, 40))
    buf = io.BytesIO()
    img.save(buf, fmt)
    return buf.getvalue()


GOOD_READING = {"readable": True, "shop": "Cafe Mocha", "final_amount": 858, "subtotal": 780,
                "taxes_and_charges": 78, "date": "14/10/2026",
                "items": ["Cappuccino", "Paneer Sandwich"]}


@pytest.fixture
def fake_vision(monkeypatch):
    """Script what the vision model 'reads'. Returns the list of calls."""
    calls = []

    def _script(reading):
        content = reading if isinstance(reading, str) else json.dumps(reading)

        def _get(*args, **kwargs):
            calls.append(1)
            return FakeToolModel(messages=iter([AIMessage(content=content)]))
        monkeypatch.setattr(vision, "get_vision_model", _get)
        return calls
    return _script


def upload(client, data, filename="receipt.png"):
    return client.post("/expenses/scan", data={"receipt": (io.BytesIO(data), filename)},
                       content_type="multipart/form-data")


# ---------- Upload validation ----------

def test_scan_page_requires_login(client):
    assert "/login" in client.get("/expenses/scan").headers["Location"]
    assert "/login" in upload(client, make_image()).headers["Location"]


def test_scan_page_and_profile_button(client, user):
    assert client.get("/expenses/scan").status_code == 200
    assert b"Scan receipt" in client.get("/profile").data


@pytest.mark.parametrize("filename, data, message", [
    ("receipt.jpg", b"this is not an image at all", b"t a readable JPG or PNG"),   # isn't (HTML-escaped)
    ("receipt.png", make_image()[:200], b"t a readable JPG or PNG"),               # truncated
    ("receipt.gif", make_image("GIF"), b"Please upload a JPG"),
    ("receipt.png", make_image("GIF"), b"Please upload a JPG"),                  # GIF renamed .png
    ("receipt.exe", make_image(), b"Please upload a JPG"),
    ("receipt.png", b"", b"Please choose a photo"),
])
def test_bad_uploads_are_rejected_without_calling_the_ai(client, user, monkeypatch, filename, data, message):
    monkeypatch.setattr(vision, "get_vision_model", lambda *a, **k: pytest.fail("AI must not be called"))
    response = upload(client, data, filename)
    assert response.status_code == 400
    assert message in response.data


def test_upload_over_5_mb_is_rejected(client, user, monkeypatch):
    monkeypatch.setattr(vision, "get_vision_model", lambda *a, **k: pytest.fail("AI must not be called"))
    noise = (np.random.default_rng(1).random((1370, 1370, 3)) * 255).astype(np.uint8)   # ~5.6 MB as PNG
    big = cv2.imencode(".png", noise)[1].tobytes()
    assert vision.MAX_UPLOAD_BYTES < len(big) < 6 * 1024 * 1024
    response = upload(client, big)
    assert response.status_code == 400 and b"under 5 MB" in response.data


def test_request_over_the_server_limit_is_refused(client, user):
    response = upload(client, b"x" * (7 * 1024 * 1024), "huge.png")
    assert response.status_code == 302                        # 413 -> friendly message + redirect
    assert b"under 5 MB" in client.get("/expenses/scan").data


# ---------- OpenCV preprocessing ----------

def test_preprocess_resizes_grayscales_and_keeps_text_readable():
    processed, info = vision.preprocess(make_image("JPEG", size=(3000, 2000)))
    image = cv2.imdecode(np.frombuffer(processed, np.uint8), cv2.IMREAD_UNCHANGED)
    assert image.ndim == 2                                    # grayscale: one channel
    assert max(image.shape) == vision.MAX_SIDE                # longest side 1600 px
    assert info == {"original_size": "3000×2000", "processed_size": "1600×1067", "resized": True}


def test_preprocess_leaves_small_images_their_size_and_boosts_contrast():
    dull = Image.new("RGB", (300, 200), (150, 150, 150))
    ImageDraw.Draw(dull).rectangle((50, 50, 250, 150), fill=(120, 120, 120))   # faint "ink"
    buf = io.BytesIO(); dull.save(buf, "PNG")
    processed, info = vision.preprocess(buf.getvalue())
    image = cv2.imdecode(np.frombuffer(processed, np.uint8), cv2.IMREAD_UNCHANGED)
    assert info["resized"] is False and image.shape == (200, 300)
    assert int(image.max()) - int(image.min()) > 30           # more contrast than the original 30


# ---------- The whole scan: preview only, saving needs Confirm ----------

def test_scan_shows_editable_preview_and_saves_nothing(client, user, fake_vision):
    calls = fake_vision(GOOD_READING)
    response = upload(client, make_image("JPEG"), "bill.jpg")
    page = response.get_data(as_text=True)
    assert response.status_code == 200 and len(calls) == 1
    assert 'name="amount" value="858.00"' in page             # final amount, not the 780 subtotal
    assert 'name="date" value="2026-10-14"' in page           # DD/MM/YYYY read day-first
    assert '<option value="Food" selected>' in page           # "Cafe" -> Food, chosen by code
    assert "Cafe Mocha — Cappuccino, Paneer Sandwich" in page
    assert "data:image/jpeg;base64," in page and "data:image/png;base64," in page   # before/after
    assert "Confirm &amp; save" in page
    assert count_expenses(user) == 0                          # nothing saved yet


def test_confirm_saves_through_the_normal_add_route(client, user, fake_vision):
    fake_vision(GOOD_READING)
    upload(client, make_image())
    # The preview form posts here; the user may have edited the values first.
    response = client.post("/expenses/add", data={"amount": "858.00", "category": "Food",
                                                  "date": "2026-10-14", "description": "Cafe Mocha"})
    assert response.status_code == 302
    expense = db.get_expenses_for_user(user)[0]
    assert (expense["amount"], expense["category"], expense["date"]) == (858.0, "Food", "2026-10-14")


def test_unreadable_receipt_lets_the_user_fill_in_the_fields(client, user, fake_vision):
    fake_vision({"readable": False, "shop": None, "final_amount": None, "date": None, "items": []})
    page = upload(client, make_image()).get_data(as_text=True)
    assert "Couldn&#39;t read this receipt clearly" in page
    assert 'name="amount" value=""' in page
    assert f'name="date" value="{TODAY.isoformat()}"' in page   # missing date -> today
    assert count_expenses(user) == 0


def test_model_reply_that_is_not_json_is_treated_as_unreadable(client, user, fake_vision):
    fake_vision("sorry, I can't see that")
    page = upload(client, make_image()).get_data(as_text=True)
    assert "Couldn&#39;t read this receipt clearly" in page


def test_ollama_down_shows_receipt_scanning_unavailable(client, user, monkeypatch):
    class Down(FakeToolModel):
        def _generate(self, *args, **kwargs):
            raise httpx.ConnectError("refused")
    monkeypatch.setattr(vision, "get_vision_model", lambda *a, **k: Down(messages=iter([])))
    response = upload(client, make_image())
    assert response.status_code == 503
    assert b"Receipt scanning unavailable" in response.data
    assert b"Add the expense manually" in response.data


# ---------- interpret(): checks done in code ----------

def test_old_receipt_gets_a_note():
    old = (TODAY - timedelta(days=400)).strftime("%d/%m/%Y")
    result = vision.interpret({**GOOD_READING, "date": old}, TODAY)
    assert any("This receipt is from" in n and "change the date if needed" in n for n in result["notes"])


def test_future_date_falls_back_to_today():
    future = (TODAY + timedelta(days=30)).strftime("%d/%m/%Y")
    result = vision.interpret({**GOOD_READING, "date": future}, TODAY)
    assert result["date"] == TODAY.isoformat() and any("in the future" in n for n in result["notes"])


def test_amount_equal_to_subtotal_with_taxes_is_flagged():
    result = vision.interpret({**GOOD_READING, "final_amount": 780, "subtotal": 780, "taxes_and_charges": 78})
    assert any("BEFORE taxes" in n for n in result["notes"])


@pytest.mark.parametrize("value, expected", [
    (858, 858.0), ("1,180.50", 1180.5), ("₹ 1180", 1180.0), ("Rs. 99/-", 99.0), ("INR 250", 250.0),
    (0, None), (-50, None), ("abc", None), (None, None), (True, None), ("12.5.3", None),
])
def test_parse_amount(value, expected):
    assert vision.parse_amount(value) == expected


@pytest.mark.parametrize("value, expected", [
    ("03/04/2026", date(2026, 4, 3)),          # Indian: 3 April, not March 4
    ("14-10-2026", date(2026, 10, 14)), ("14.10.26", date(2026, 10, 14)),
    ("14/10/2026 13:42", date(2026, 10, 14)), ("12th Oct 2026", date(2026, 10, 12)),
    ("04/11/2026, 09:12 PM", date(2026, 11, 4)),                 # UPI screenshot style (found in a real run)
    ("14-10-2026 | 1:05:09 pm", date(2026, 10, 14)), ("12 Oct 2026 at 9:30", date(2026, 10, 12)),
    ("2026-10-14", date(2026, 10, 14)), ("Oct 14, 2026", date(2026, 10, 14)),
    ("31/02/2026", None), ("yesterday", None), ("", None), (None, None),
])
def test_parse_receipt_date(value, expected):
    assert vision.parse_receipt_date(value) == expected


@pytest.mark.parametrize("shop, items, expected", [
    ("Cafe Coffee Day", ["Latte"], "Food"), ("Uber", [], "Transport"), ("Apollo Pharmacy", ["Dolo 650"], "Health"),
    ("PVR Cinemas", ["Popcorn"], "Entertainment"), ("Airtel", ["Prepaid recharge"], "Bills"),
    ("Amazon", ["Shoes"], "Shopping"), ("Ramesh", ["Misc"], "Other"),
    ("Steakhouse current", ["emission"], "Other"),  # whole words only: not "tea", "rent", "emi"
    # No business type: item words decide (the bug: a restaurant name with no "restaurant" in it)
    ("The Local Diners", ["Grilled Chicken", "Caesar Salad", "Virgin Mojito"], "Food"),
    ("Sharma & Sons", ["Butter Naan", "Paneer Tikka"], "Food"),
    ("Sri Balaji Agencies", ["Crocin 650", "Cough Syrup"], "Health"),
    ("Star Health and Allied Insurance", ["Renewal premium"], "Health"),   # health insurance -> Health
    ("Mahadev Properties", ["Rent for November"], "Bills"),
])
def test_pick_category_from_shop_and_items(shop, items, expected):
    assert vision.pick_category(shop, items) == expected


@pytest.mark.parametrize("business_type, expected", [
    ("restaurant", "Food"), ("food_delivery", "Food"), ("grocery", "Food"),
    ("pharmacy", "Health"), ("hospital", "Health"), ("diagnostic_lab", "Health"), ("health_insurance", "Health"),
    ("insurance", "Bills"), ("electricity", "Bills"), ("water", "Bills"), ("gas", "Bills"),
    ("mobile_internet_dth", "Bills"), ("rent", "Bills"),
    ("fuel", "Transport"), ("cab", "Transport"), ("travel_ticket", "Transport"),
    ("online_shopping", "Shopping"), ("retail_store", "Shopping"), ("entertainment", "Entertainment"),
    # how the model might write it
    ("Online Shopping", "Shopping"), ("cafe", "Food"), ("utility", "Bills"), ("telecom", "Bills"),
    ("Petrol", "Transport"), ("OTT", "Entertainment"),
])
def test_business_type_decides_the_category(business_type, expected):
    assert vision.pick_category("Some Business", ["thing"], business_type) == expected


def test_the_local_diners_is_food_whatever_the_model_says():
    items = ["Chicken", "Salad", "Mojito"]
    assert vision.pick_category("The Local Diners", items, "restaurant") == "Food"
    assert vision.pick_category("The Local Diners", items, "other") == "Food"          # falls back to items
    assert vision.pick_category("The Local Diners", items, "spaceship") == "Food"      # invalid type ignored
    assert vision.pick_category("The Local Diners", items, None) == "Food"


def test_insurance_type_with_health_words_is_health():
    assert vision.pick_category("Niva Bupa", ["Health policy premium"], "insurance") == "Health"
    assert vision.pick_category("LIC of India", ["Term policy premium"], "insurance") == "Bills"


def test_business_type_is_shown_and_used_in_the_preview(client, user, fake_vision):
    fake_vision({**GOOD_READING, "shop": "The Local Diners", "business_type": "restaurant",
                 "items": ["Chicken", "Salad", "Mojito"]})
    page = upload(client, make_image()).get_data(as_text=True)
    assert '<option value="Food" selected>' in page
    assert "Detected: <strong>Restaurant</strong>" in page


# ---------- PDF bills ----------

def make_pdf(text="THE LOCAL DINERS\nGrand Total Rs 1,240.00", password=None):
    import pymupdf
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 90), text, fontsize=16)
    if password:
        data = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw=password, user_pw=password)
    else:
        data = doc.tobytes()
    doc.close()
    return data


def test_pdf_first_page_becomes_an_image():
    png = vision.render_pdf_first_page(make_pdf())
    image = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    assert image is not None and max(image.shape[:2]) == pytest.approx(vision.PDF_RENDER_SIDE, abs=2)


def test_pdf_upload_is_read_like_a_photo(client, user, fake_vision):
    calls = fake_vision({**GOOD_READING, "shop": "Tata Power", "business_type": "electricity",
                         "final_amount": 1240, "subtotal": None, "taxes_and_charges": None,
                         "date": "05/11/2026", "items": ["Electricity charges"]})
    response = upload(client, make_pdf(), "bill.pdf")
    page = response.get_data(as_text=True)
    assert response.status_code == 200 and len(calls) == 1
    assert 'name="amount" value="1240.00"' in page
    assert '<option value="Bills" selected>' in page
    assert "PDF, page 1" in page
    assert count_expenses(user) == 0


@pytest.mark.parametrize("filename, data, message", [
    ("bill.pdf", make_image(), b"t a readable PDF"),                 # an image renamed .pdf
    ("bill.png", make_pdf(), b"t a readable JPG or PNG"),            # a PDF renamed .png
    ("bill.pdf", b"%PDF-1.7 broken nonsense", b"t a readable PDF"),
    ("bill.pdf", make_pdf(password="secret"), b"password-protected"),
])
def test_bad_pdfs_are_rejected(client, user, monkeypatch, filename, data, message):
    monkeypatch.setattr(vision, "get_vision_model", lambda *a, **k: pytest.fail("AI must not be called"))
    response = upload(client, data, filename)
    assert response.status_code == 400
    assert message in response.data


def test_pdf_over_5_mb_is_rejected(client, user, monkeypatch):
    monkeypatch.setattr(vision, "get_vision_model", lambda *a, **k: pytest.fail("AI must not be called"))
    big = b"%PDF-1.7\n" + b"0" * (vision.MAX_UPLOAD_BYTES + 10)
    response = upload(client, big, "big.pdf")
    assert response.status_code == 400 and b"under 5 MB" in response.data


def test_prompt_asks_for_final_amount_and_bill_date():
    prompt = vision.READ_PROMPT
    for label in ["Net Amount", "Grand Total", "Amount Paid", "Total Due", "Premium Amount"]:
        assert label in prompt
    for wrong in ["Gross Total", "Subtotal", "Arrears"]:
        assert f"NEVER use {wrong}" in prompt or wrong in prompt.split("NEVER use", 1)[1]
    assert "NEVER use the due date" in prompt
    assert all(f'"{t}"' in prompt for t in vision.BUSINESS_TYPES)
