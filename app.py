# app.py - Dyslexia detection web app (updated sensitivity + fallback rule)
import io
import os
from pathlib import Path
from flask import Flask, request, render_template_string, jsonify
from PIL import Image, ImageOps, ImageDraw
import pytesseract
import numpy as np

from tensorflow.keras.models import load_model
from difflib import SequenceMatcher
try:
    import enchant
    DICT = enchant.Dict("en_US")
except Exception:
    DICT = None

# ---------------- CONFIG ----------------
MODEL_PATH = "notebook/safe_glyphnet_rebalanced.keras"  # use fine-tuned model if available
CLASS_NAMES = ["Corrected", "Normal", "Reversal"]       # model index -> label
CHAR_PAD = 12
WORD_PAD = 18
THRESH_WORD_REVERSAL_RATIO = 0.3

# decision tuning (more sensitive as requested)
MIN_CONF_TO_COUNT_REVERSAL = 0.65   # count reversals with >= 65% model confidence
MIN_REVERSED_CHARS_ABSOLUTE = 6     # need at least 6 counted reversals
MIN_CHARS_TO_EVALUATE = 25          # require >= 25 chars to evaluate reliably
WEIGHTED_REVERSAL_THRESHOLD = 0.06  # lower weighted threshold

DEBUG_DIR = Path("debug_crops")
STATIC_DIR = Path("static")
DEBUG_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

app = Flask(__name__)

# ---------------- load model ----------------
model = None
TARGET_W, TARGET_H = 96, 96
try:
    model = load_model(MODEL_PATH)
    ishape = model.input_shape
    if ishape and len(ishape) >= 3 and ishape[1] and ishape[2]:
        TARGET_H = int(ishape[1]); TARGET_W = int(ishape[2])
    print("Loaded model:", MODEL_PATH)
    print("Target size:", TARGET_W, "x", TARGET_H)
except Exception as e:
    print("Warning: could not load model", MODEL_PATH, ":", e)
    model = None

# ---------------- helpers ----------------
def cv_clean_image(pil_img):
    """Stronger denoising + adaptive threshold for messy scans (uses OpenCV if available)."""
    try:
        import cv2 as _cv2
        import numpy as _np
        arr = _cv2.cvtColor(_np.array(pil_img), _cv2.COLOR_RGB2GRAY)
        arr = _cv2.fastNlMeansDenoising(arr, None, h=10, templateWindowSize=7, searchWindowSize=21)
        arr = _cv2.GaussianBlur(arr, (3,3), 0)
        th = _cv2.adaptiveThreshold(arr, 255, _cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   _cv2.THRESH_BINARY, 15, 9)
        if th.mean() < 127:
            th = 255 - th
        kernel = _cv2.getStructuringElement(_cv2.MORPH_RECT, (2,2))
        th = _cv2.morphologyEx(th, _cv2.MORPH_OPEN, kernel)
        return Image.fromarray(th)
    except Exception:
        out = ImageOps.autocontrast(pil_img.convert("L"), cutoff=2)
        w,h = out.size
        if min(w,h) < 512:
            out = out.resize((int(w*1.2), int(h*1.2)), Image.Resampling.LANCZOS)
        return out

def tbox_to_pil_box(parts, image_h):
    x1 = int(parts[1]); y1 = int(parts[2]); x2 = int(parts[3]); y2 = int(parts[4])
    top = image_h - y2
    bottom = image_h - y1
    return (x1, top, x2, bottom)

def preprocess_crop_for_model(pil_crop):
    crop = pil_crop.convert("L")
    crop = ImageOps.autocontrast(crop, cutoff=1)
    w,h = crop.size
    # scale to fit target while preserving aspect
    ratio = min(TARGET_W / max(1,w), TARGET_H / max(1,h))
    new_w = max(1, int(w * ratio)); new_h = max(1, int(h * ratio))
    crop = crop.resize((new_w, new_h), Image.Resampling.LANCZOS)
    pad_left = (TARGET_W - new_w) // 2
    pad_top = (TARGET_H - new_h) // 2
    pad_right = TARGET_W - new_w - pad_left
    pad_bottom = TARGET_H - new_h - pad_top
    crop = ImageOps.expand(crop, border=(pad_left, pad_top, pad_right, pad_bottom), fill=255)
    arr = np.asarray(crop).astype(np.float32) / 255.0
    arr = arr.reshape((1, TARGET_H, TARGET_W, 1))
    return arr

def predict_on_crop(pil_crop):
    if model is None:
        return "Normal", 1.0
    X = preprocess_crop_for_model(pil_crop)
    probs = model.predict(X, verbose=0)[0]
    idx = int(np.argmax(probs))
    label = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"IDX{idx}"
    return label, float(probs[idx])

def safe_crop(img, box, pad=0):
    W, H = img.size
    x1, y1, x2, y2 = box
    x1c = max(0, int(x1 - pad)); y1c = max(0, int(y1 - pad))
    x2c = min(W, int(x2 + pad)); y2c = min(H, int(y2 + pad))
    if x2c <= x1c or y2c <= y1c:
        cx = int((x1 + x2) / 2); cy = int((y1 + y2) / 2)
        return img.crop((max(0, cx-12), max(0, cy-12), min(W, cx+12), min(H, cy+12)))
    return img.crop((x1c, y1c, x2c, y2c))

def clean_candidate_text(s):
    if not s: return None
    s = s.strip().replace("\n", " ").strip()
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '-.,;:()?")
    if all((ch in allowed) for ch in s):
        return s
    return None

def is_plausible_word(orig, cand):
    if not cand: return False
    cand_token = cand.split()[0] if cand.split() else cand
    if DICT:
        return DICT.check(cand_token) or DICT.check(cand_token.lower())
    sim = SequenceMatcher(None, orig.lower(), cand_token.lower()).ratio()
    return sim >= 0.65 and abs(len(orig) - len(cand_token)) <= max(1, len(orig)//2)

# ---------------- templates ----------------
INDEX_HTML = """
<!doctype html>
<title>Dyslexia Detection</title>
<h2>Upload paragraph image</h2>
<form method=post enctype=multipart/form-data action="/analyze">
  <input type=file name=file accept="image/*" required>
  <input type=submit value="Analyze">
</form>
<p>Model: <b>{}</b> | Target crop size: <b>{}×{}</b></p>
""".format(MODEL_PATH, TARGET_W, TARGET_H)

RESULT_HTML = """
<!doctype html>
<title>Result</title>
<h2>Result</h2>
<p><b>Verdict:</b> {{ verdict }}</p>
<p><b>Total characters:</b> {{ total_chars }} | <b>Reversed chars:</b> {{ reversed_chars }} ({{ percent }}%)</p>
<h3>Original paragraph (OCR):</h3>
<pre>{{ original_text }}</pre>
<h3>Corrected paragraph (after safe auto-corrections):</h3>
<pre>{{ corrected_text }}</pre>
{% if image_url %}
<h3>Annotated image</h3>
<img src="{{ image_url }}" style="max-width:100%; border:1px solid #ddd;">
<ul>
<li>Red boxes = flagged reversed characters</li>
<li>Blue boxes = words auto-corrected</li>
</ul>
{% endif %}
<hr>
<a href="/">Analyze another</a>
"""

# ---------------- routes ----------------
@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML)

@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return "No file uploaded", 400
    file = request.files["file"]
    img_bytes = file.read()
    img_rgb = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    W, H = img_rgb.size

    cleaned_for_ocr = cv_clean_image(img_rgb)

    OCR_PSM = 6
    ocr_conf = f"--oem 3 --psm {OCR_PSM}"

    data = pytesseract.image_to_data(cleaned_for_ocr.convert("RGB"), config=ocr_conf, output_type=pytesseract.Output.DICT)
    words = []
    for i in range(len(data['text'])):
        txt = data['text'][i].strip()
        if txt == "":
            continue
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        buf = 6
        words.append({"text": txt, "box": (x-buf, y-buf, x+w+buf, y+h+buf), "index": i})

    boxes_str = pytesseract.image_to_boxes(cleaned_for_ocr.convert("RGB"), config=ocr_conf)
    per_char_info = []
    for line in boxes_str.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        pil_box = tbox_to_pil_box(parts, H)
        ch = parts[0]
        per_char_info.append({"char": ch, "box": pil_box})

    if len(per_char_info) == 0:
        return "No characters detected. Try a clearer image.", 400

    # tolerant mapping char -> word
    for w in words:
        w['chars'] = []
    for cinfo in per_char_info:
        cx1, cy1, cx2, cy2 = cinfo['box']
        cx = (cx1 + cx2) / 2
        cy = (cy1 + cy2) / 2
        assigned = False
        for w in words:
            wx1, wy1, wx2, wy2 = w['box']
            margin = 8
            if (cx >= wx1 - margin) and (cx <= wx2 + margin) and (cy >= wy1 - margin) and (cy <= wy2 + margin):
                w['chars'].append({**cinfo, "center": (cx, cy)})
                assigned = True
                break
        if not assigned:
            best=None; best_dist=1e9
            for w in words:
                wx1, _, wx2, _ = w['box']
                dist = min(abs(cx - wx1), abs(cx - wx2))
                if dist < best_dist:
                    best_dist = dist; best = w
            if best:
                best['chars'].append({**cinfo, "center": (cx, cy)})

    annotated = img_rgb.copy()
    draw = ImageDraw.Draw(annotated)

    for widx, w in enumerate(words):
        w['chars'].sort(key=lambda c: c.get('center', (0,0))[0])
        word_reversed_chars = 0
        for ci, cinfo in enumerate(w['chars']):
            cx1, cy1, cx2, cy2 = cinfo['box']
            char_crop = safe_crop(img_rgb, (cx1, cy1, cx2, cy2), pad=CHAR_PAD)
            if char_crop.size[0] < 12 or char_crop.size[1] < 12:
                char_crop = safe_crop(img_rgb, w['box'], pad=WORD_PAD)
            char_crop_l = char_crop.convert("L")

            label, conf = predict_on_crop(char_crop_l)
            cinfo['pred_label'] = label
            cinfo['pred_conf'] = float(conf)

            try:
                if (label == "Reversal") or (float(conf) < MIN_CONF_TO_COUNT_REVERSAL):
                    DEBUG_DIR.mkdir(exist_ok=True)
                    fname = DEBUG_DIR / f"w{widx:03d}_c{ci:03d}_{label}_{conf:.3f}.png"
                    char_crop_l.resize((64,64)).save(fname)
            except Exception:
                pass

            if label == "Reversal":
                word_reversed_chars += 1
                draw.rectangle([cx1, cy1, cx2, cy2], outline="red", width=2)
            else:
                draw.rectangle([cx1, cy1, cx2, cy2], outline="green", width=1)

        ratio = word_reversed_chars / max(1, len(w['chars']))
        if ratio >= THRESH_WORD_REVERSAL_RATIO:
            wx1, wy1, wx2, wy2 = w['box']
            wx1c = max(0, wx1 - WORD_PAD); wy1c = max(0, wy1 - WORD_PAD)
            wx2c = min(W, wx2 + WORD_PAD); wy2c = min(H, wy2 + WORD_PAD)
            word_crop = img_rgb.crop((wx1c, wy1c, wx2c, wy2c))
            flipped = ImageOps.mirror(word_crop)
            candidate = pytesseract.image_to_string(flipped, config="--psm 7").strip()
            candidate = clean_candidate_text(candidate)
            if candidate and is_plausible_word(w['text'], candidate):
                draw.rectangle([wx1c, wy1c, wx2c, wy2c], outline="blue", width=3)

    original_text = " ".join([w['text'] for w in words])
    corrected_text = " ".join([w.get('correction', w['text']) for w in words])

    # ---------- improved decision logic & debugging output (updated) ----------
    per_char_records = []
    for widx, w in enumerate(words):
        for ci, cinfo in enumerate(w.get('chars', [])):
            lbl = cinfo.get('pred_label') or cinfo.get('label') or "Unknown"
            conf = float(cinfo.get('pred_conf', cinfo.get('conf', 0.0)))
            per_char_records.append({
                'word_idx': widx,
                'char_idx': ci,
                'char': cinfo.get('char', '?'),
                'label': lbl,
                'conf': conf,
                'box': cinfo.get('box')
            })

    total_char_count = len(per_char_records)
    # count how many chars were visually flagged as reversed (any pred_label == "Reversal")
    visual_reversed_count = sum(1 for r in per_char_records if r['label'] == "Reversal")
    percent_reversed_display = (visual_reversed_count / max(1, total_char_count))

    # count only high-confidence reversals for the 'counted' metric
    counted_reversals = 0
    weighted_reversal_sum = 0.0
    low_conf_count = 0

    for rec in per_char_records:
        if rec['label'] == "Reversal":
            if rec['conf'] >= MIN_CONF_TO_COUNT_REVERSAL:
                counted_reversals += 1
                weighted_reversal_sum += rec['conf']
            else:
                low_conf_count += 1

    weighted_score = weighted_reversal_sum / max(1.0, float(total_char_count))

    verdict = "✅ No Dyslexia Detected"
    notes = []

    if total_char_count < MIN_CHARS_TO_EVALUATE:
        verdict = "⚠️ Uncertain — too few characters to decide reliably"
        notes.append(f"Only {total_char_count} chars")
    else:
        # primary rule: enough high-confidence reversals and weighted_score
        if (counted_reversals >= MIN_REVERSED_CHARS_ABSOLUTE) and (weighted_score >= WEIGHTED_REVERSAL_THRESHOLD):
            verdict = "⚠️ Possible Dyslexia Detected"
            notes.append(f"counted_reversals={counted_reversals}, weighted_score={weighted_score:.3f}")
        else:
            # fallback: if a large fraction of chars are visually flagged reversed (even if low-conf)
            if percent_reversed_display >= 0.25 and visual_reversed_count >= 6:
                verdict = "⚠️ Possible Dyslexia Detected"
                notes.append(f"visual_reversed={visual_reversed_count}/{total_char_count} ({percent_reversed_display:.2%}) — fallback triggered")
            else:
                verdict = "✅ No Dyslexia Detected"
                notes.append(f"counted_reversals={counted_reversals}, weighted_score={weighted_score:.3f}, low_conf_reversals={low_conf_count}, visual_reversed={visual_reversed_count}")

    try:
        import csv
        csvp = DEBUG_DIR / "predictions.csv"
        with open(csvp, "w", newline="", encoding="utf8") as cf:
            writer = csv.DictWriter(cf, fieldnames=['word_idx','char_idx','char','label','conf','box'])
            writer.writeheader()
            for rec in per_char_records:
                writer.writerow(rec)
        notes.append(f"Saved per-char CSV to {str(csvp)}")
    except Exception:
        pass

    if notes:
        verdict = verdict + " — " + " | ".join(notes)
    # ---------- end decision logic ----------

    percent_reversed = (visual_reversed_count / max(1, total_char_count)) * 100.0 if total_char_count > 0 else 0.0

    out_name = f"annotated_{os.getpid()}.jpg"
    out_path = STATIC_DIR / out_name
    annotated.save(out_path)

    return render_template_string(RESULT_HTML,
                                  verdict=verdict,
                                  total_chars=total_char_count,
                                  reversed_chars=visual_reversed_count,
                                  percent=round(percent_reversed, 2),
                                  original_text=original_text,
                                  corrected_text=corrected_text,
                                  image_url=f"/static/{out_name}")

@app.route("/predict_json", methods=["POST"])
def predict_json():
    if "file" not in request.files:
        return jsonify({"error":"no file"}), 400
    file = request.files["file"]
    img = Image.open(io.BytesIO(file.read())).convert("RGB")
    cleaned = cv_clean_image(img)
    W, H = img.size

    OCR_PSM = 6
    ocr_conf = f"--oem 3 --psm {OCR_PSM}"
    data = pytesseract.image_to_data(cleaned.convert("RGB"), config=ocr_conf, output_type=pytesseract.Output.DICT)
    words = []
    for i in range(len(data['text'])):
        txt = data['text'][i].strip()
        if txt == "": continue
        x,y,w,h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        buf=6
        words.append({"text":txt,"box":[x-buf,y-buf,x+w+buf,y+h+buf]})

    boxes_str = pytesseract.image_to_boxes(cleaned.convert("RGB"), config=ocr_conf)
    per_char = []
    for line in boxes_str.splitlines():
        parts = line.split()
        if len(parts) < 5: continue
        box = tbox_to_pil_box(parts, H)
        crop = img.crop(box).convert("L")
        label, conf = predict_on_crop(crop)
        per_char.append({"char": parts[0], "box": list(box), "label": label, "conf": float(conf)})

    return jsonify({"words": words, "per_char": per_char})

# ---------------- run ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


