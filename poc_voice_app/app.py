"""
Poshan AI Voice Assistant — POC v3.0
• Sarvam STT (saaras:v3) · TTS (bulbul:v1) · Translate (mayura:v1)
• FAISS RAG on Poshan Tracker knowledge base
• Language-matched responses (answer in same language as question)
• On-demand translation toggle (any language ⇄ English)
• Fixed field extraction: word-based decimals ("आठ पॉइंट दो" → 8.2)
                          proximity age matching ("चौदह महीने" → 14)
• 12 languages: English + 11 scheduled Indian languages
• Conversation Q&A format: question shown, then answer in same language
"""

import asyncio, base64, os, re, subprocess, tempfile, time
import numpy as np
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from collections import deque
from poshan_kb import DOCS

SARVAM_KEY  = os.getenv("SARVAM_API_KEY", "sk_l6rldfif_gSyfCcpZP0RXcyJNlIgb15Vr")
SAMPLE_RATE = 16000
HISTORY     = deque(maxlen=20)

app = FastAPI(title="Poshan AI POC v3")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ── RAG — manual KB ───────────────────────────────────────────────────────────
import json as _json
from pathlib import Path as _Path

print("[startup] Loading embedding model + pre-built FAISS index…")
t0 = time.time()
_MANUAL_INDEX_PATH = _Path(__file__).parent / "manual_faiss.index"
_MANUAL_TEXTS_PATH = _Path(__file__).parent / "manual_texts.json"
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    _emb  = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    _docs_data = _json.loads(_MANUAL_TEXTS_PATH.read_text(encoding="utf-8"))
    _texts     = [f"{d['title']}\n{d['content']}" for d in _docs_data]
    _idx       = faiss.read_index(str(_MANUAL_INDEX_PATH))
    RAG_OK     = True
    print(f"[startup] Manual RAG ready in {time.time()-t0:.1f}s — {_idx.ntotal} vectors")
except Exception as e:
    print(f"[startup] RAG unavailable: {e}")
    RAG_OK = False

# ── RAG — PDF knowledge base (pre-built by pdf_ingest.py) ────────────────────
_PDF_INDEX_PATH  = _Path(__file__).parent / "pdf_faiss.index"
_PDF_CHUNKS_PATH = _Path(__file__).parent / "pdf_chunks.json"

print("[startup] Loading PDF FAISS index…")
try:
    _pdf_chunks: list[dict] = _json.loads(_PDF_CHUNKS_PATH.read_text(encoding="utf-8"))
    _pdf_idx    = faiss.read_index(str(_PDF_INDEX_PATH))
    PDF_RAG_OK  = True
    print(f"[startup] PDF RAG ready — {len(_pdf_chunks)} chunks, {_pdf_idx.ntotal} vectors")
except Exception as e:
    print(f"[startup] PDF RAG unavailable: {e}")
    PDF_RAG_OK  = False
    _pdf_chunks = []
    _pdf_idx    = None

_rag_cache: dict = {}


def rag_search(query: str, k: int = 5) -> list[dict]:
    if not RAG_OK or not query.strip():
        return []
    cache_key = f"{normalize(query)}:{k}"
    if cache_key in _rag_cache:
        return _rag_cache[cache_key]

    q_vec = _emb.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_vec)

    # ── manual docs ───────────────────────────────────────────────────────
    scores_m, idxs_m = _idx.search(q_vec, k)
    manual = [
        {"id": DOCS[i]["id"], "title": DOCS[i]["title"],
         "snippet": DOCS[i]["content"].strip()[:400] + "…",
         "full": DOCS[i]["content"].strip(),
         "source": "Poshan KB", "page": None,
         "score": round(float(s), 3)}
        for s, i in zip(scores_m[0], idxs_m[0]) if i >= 0
    ]

    # ── PDF docs ──────────────────────────────────────────────────────────
    pdf = []
    if PDF_RAG_OK and _pdf_idx is not None:
        scores_p, idxs_p = _pdf_idx.search(q_vec, k)
        seen_sources: set = set()
        for s, i in zip(scores_p[0], idxs_p[0]):
            if i < 0 or float(s) < 0.35:
                continue
            c = _pdf_chunks[i]
            key = c["source"]
            if key in seen_sources:
                continue
            seen_sources.add(key)
            pdf.append({
                "id": c["id"], "title": c["title"],
                "snippet": c["content"][:400] + "…",
                "full": c["content"],
                "source": c["source"], "page": c["page"],
                "score": round(float(s), 3),
            })

    # ── merge: interleave manual (higher trust) + PDF ─────────────────────
    combined = []
    seen_ids: set = set()
    for r in manual + sorted(pdf, key=lambda x: -x["score"]):
        if r["id"] not in seen_ids:
            seen_ids.add(r["id"])
            combined.append(r)

    result = combined[:k]
    if len(_rag_cache) < 512:
        _rag_cache[cache_key] = result
    return result


# ── WHO NORMS ─────────────────────────────────────────────────────────────────
WHO = {0:(2.5,4.5),3:(4.5,7.0),6:(5.5,9.0),9:(6.5,10.5),12:(7.0,11.5),
       14:(7.5,11.8),18:(7.5,13.0),24:(9.0,15.0),36:(10.5,17.0),
       48:(12.0,19.5),60:(13.5,22.0)}

def who_norm(age_m: int) -> tuple:
    closest = min(WHO, key=lambda x: abs(x - age_m))
    return WHO[closest]

def nutrition_status(weight: float, age_m: int) -> dict:
    lo, hi = who_norm(age_m)
    if weight < lo - 0.8:
        return {"label":"SAM","color":"red","action":"Refer to NRC immediately"}
    if weight < lo:
        return {"label":"MAM","color":"orange","action":"Provide RUTF therapeutic food"}
    if weight > hi:
        return {"label":"Overweight","color":"purple","action":"Dietary counselling recommended"}
    return {"label":"Normal","color":"green","action":"Continue monthly monitoring"}


# ── HELPERS ───────────────────────────────────────────────────────────────────
def normalize(text: str) -> str:
    """Strip nuktas/diacritics so वज़न matches वजन."""
    return re.sub(r'[़‌‍]', '', text).lower().strip()

def sarvam_lc(l: str) -> str:
    """Short code → Sarvam API language code."""
    return f"{l}-IN"

NUM_HI = {
    "शून्य":0,"एक":1,"दो":2,"तीन":3,"चार":4,"पांच":5,"छह":6,"सात":7,
    "आठ":8,"नौ":9,"दस":10,"ग्यारह":11,"बारह":12,"तेरह":13,"चौदह":14,
    "पंद्रह":15,"सोलह":16,"सत्रह":17,"अठारह":18,"उन्नीस":19,"बीस":20,
    "तीस":30,"चालीस":40,"पचास":50,"साठ":60,
    "ek":1,"do":2,"teen":3,"char":4,"paanch":5,"chhe":6,"saat":7,"aath":8,
    "nau":9,"das":10,"gyarah":11,"barah":12,"terah":13,"chaudah":14,
    "choda":14,"pandrah":15,"solah":16,"bees":20,"tees":30,
}


# ── SARVAM TRANSLATE ──────────────────────────────────────────────────────────
def sarvam_translate(text: str, source: str = "en", target: str = "hi") -> str:
    """Translate via Sarvam mayura:v1. Returns original text on failure."""
    if not text.strip() or source == target:
        return text
    try:
        r = requests.post(
            "https://api.sarvam.ai/translate",
            headers={"api-subscription-key": SARVAM_KEY,
                     "Content-Type": "application/json"},
            json={
                "input": text[:500],
                "source_language_code": sarvam_lc(source),
                "target_language_code": sarvam_lc(target),
                "speaker_gender": "Female",
                "mode": "formal",
                "model": "mayura:v1",
                "enable_preprocessing": False,
            },
            timeout=15,
        )
        if r.status_code == 200:
            return r.json().get("translated_text", text)
    except Exception:
        pass
    return text


def generate_text_answer(answer_dict: dict) -> str:
    """Strip HTML from answer dict → plain translatable text."""
    if not answer_dict:
        return ""
    title   = answer_dict.get("title", "")
    content = re.sub(r'<[^>]+>', ' ', answer_dict.get("content", ""))
    content = re.sub(r'Source:.*', '', content)
    content = re.sub(r'\s+', ' ', content).strip()
    return f"{title}. {content}"


# ── QUESTION DETECTION ────────────────────────────────────────────────────────
Q_MARKERS = [
    "?","क्या","कितना","कितनी","कैसे","कब","कहाँ","क्यों","बताओ",
    "what","how","when","where","which","why","should","chahiye","tell me",
    "होना चाहिए","होती है","होता है","hona chahiye","explain","normal","range",
    "underweight","overweight","obese","obesity","bmi","fit","healthy","fitness",
    "is he","is she","is it","is this","or he","or she","tell me","advise","advice",
    "theek","sahi","theek hai","sahi hai","kitna hona","normal hai","check",
]

def is_question(t: str) -> bool:
    tl = normalize(t)
    return any(m in tl for m in Q_MARKERS)


# ── WORD → AGE HELPER (used by rag_answer) ────────────────────────────────────
def word_to_age_months(text: str):
    t = normalize(text)
    am = re.search(r'(\d+)\s*(?:महीने|mahine|month)', t)
    ay = re.search(r'(\d+)\s*(?:साल|saal|year)', t)
    if am: return int(am.group(1))
    if ay: return int(ay.group(1)) * 12
    for word, num in NUM_HI.items():
        if re.search(rf'\b{re.escape(word)}\b\s+(?:\w+\s+)?(?:महीने|mahine|month)', t):
            return num
    return None


# ── FIELD EXTRACTION ──────────────────────────────────────────────────────────
def extract_fields(text: str) -> dict:
    fields = {}
    tn = normalize(text)      # nukta-stripped + lowercased for all matching

    # ── NAME ─────────────────────────────────────────────────────────────
    STOP_WORDS = {"height","weight","age","muac","village","child","name","body",
                  "foot","feet","inch","kilo","gram","month","year","cm","kg"}
    for pat in [
        r'(?:my\s+child|my\s+(?:son|daughter|baby))\s*(?:\'s)?\s+name\s+is\s+([A-Za-z]{2,20})',
        r'(?:naam|name)\s+(?:hai\s+|is\s+)([A-Za-z]{2,20})',
        r'([A-Za-zऀ-ॿ]{2,15})\s+(?:का|की|ke|ka)\s+(?:वजन|उम्र|नाम|wajan|umar)',
        r'(?:बच्चे?\s*का\s*नाम|naam)\s+(?:है\s+)?([A-Za-zऀ-ॿ]{2,15})',
        r'([A-Za-z]{3,15})\s+weighs\b',
    ]:
        m = re.search(pat, tn, re.IGNORECASE)
        if m:
            g = next((m.group(i) for i in range(1, (m.lastindex or 0) + 1) if m.group(i)), "")
            g = g.strip().lower()
            if g and g not in STOP_WORDS:
                fields["child_name"] = {"value": g.title(), "confidence": 0.9}
                break

    # ── WEIGHT — digit form ───────────────────────────────────────────────
    for pat in [
        r'(\d+)\s*(?:पॉइंट|point|\.)\s*(\d+)\s*(?:किलो|kilo|kg)',
        r'(?:वजन|wajan|weight)[^\d]*(\d+(?:\.\d+)?)\s*(?:किलो|kilo|kg)',
        r'(\d+(?:\.\d+)?)\s*(?:किलो|kilo|kg)',
        r'(?:body\s+)?weight\s+(?:is\s+)?(\d+(?:\.\d+)?)',
        r'weighs?\s+(\d+(?:\.\d+)?)',
    ]:
        m = re.search(pat, tn, re.IGNORECASE)
        if m:
            val = (float(f"{m.group(1)}.{m.group(2)}")
                   if m.lastindex == 2 else float(m.group(1)))
            if 1.0 < val < 300:   # sanity check (kg range)
                fields["weight_kg"] = {"value": val, "confidence": 0.9}
                break

    # ── WEIGHT — word form: "आठ पॉइंट दो किलो" → 8.2 ───────────────────
    if "weight_kg" not in fields:
        has_point = bool(re.search(r'(?:पॉइंट|point)', tn, re.IGNORECASE))
        for w1, n1 in NUM_HI.items():
            if has_point:
                m = re.search(
                    rf'\b{re.escape(w1)}\b\s+(?:पॉइंट|point)\s+(\w+)\s+(?:किलो|kilo|kg)',
                    tn, re.IGNORECASE)
                if m:
                    n2 = NUM_HI.get(m.group(1), None)
                    if n2 is not None:
                        fields["weight_kg"] = {"value": float(f"{n1}.{n2}"), "confidence": 0.9}
                        break
            else:
                m2 = re.search(
                    rf'\b{re.escape(w1)}\b\s+(?:किलो|kilo|kg)',
                    tn, re.IGNORECASE)
                if m2:
                    fields["weight_kg"] = {"value": float(n1), "confidence": 0.7}
                    break

    # ── AGE — digit form ──────────────────────────────────────────────────
    for pat in [
        r'(?:उम्र|umar|age)[^\d]*(\d+)\s*(?:महीने|mahine|months?)',
        r'(\d+)\s*(?:महीने|mahine|months?)',
    ]:
        m = re.search(pat, tn, re.IGNORECASE)
        if m:
            fields["age_months"] = {"value": int(m.group(1)), "confidence": 0.95}
            break

    # ── AGE — word form with proximity ("चौदह महीने" → 14) ───────────────
    if "age_months" not in fields:
        for word, num in NUM_HI.items():
            if re.search(
                rf'\b{re.escape(word)}\b\s+(?:\w+\s+)?(?:महीने|mahine|months?)',
                tn, re.IGNORECASE
            ):
                fields["age_months"] = {"value": num, "confidence": 0.8}
                break

    # ── AGE — year form ───────────────────────────────────────────────────
    if "age_months" not in fields:
        m = re.search(r'(\d+)\s*(?:साल|saal|years?)', tn, re.IGNORECASE)
        if m:
            fields["age_months"] = {
                "value": int(m.group(1)) * 12, "confidence": 0.9,
                "note": f"{m.group(1)} years"
            }

    # ── HEIGHT — cm ───────────────────────────────────────────────────────
    m = re.search(r'(\d+(?:\.\d+)?)\s*(?:सेंटीमीटर|cm|centimeter)', tn, re.IGNORECASE)
    if m:
        fields["height_cm"] = {"value": float(m.group(1)), "confidence": 0.95}

    # ── HEIGHT — feet + inches  e.g. "5 foot 8 inch" / "5'8"" ───────────
    if "height_cm" not in fields:
        m = re.search(
            r'(\d+)\s*(?:foot|feet|ft|\')\s*(?:and\s+)?(\d+)\s*(?:inch|inches?|in\b|")',
            tn, re.IGNORECASE)
        if m:
            cm = round(int(m.group(1)) * 30.48 + int(m.group(2)) * 2.54, 1)
            fields["height_cm"] = {"value": cm, "confidence": 0.9,
                                   "note": f"{m.group(1)}'{m.group(2)}\" → {cm} cm"}
    if "height_cm" not in fields:
        m = re.search(r'(\d+)\s*(?:foot|feet|ft)\b', tn, re.IGNORECASE)
        if m:
            cm = round(int(m.group(1)) * 30.48, 1)
            fields["height_cm"] = {"value": cm, "confidence": 0.8,
                                   "note": f"{m.group(1)} ft → {cm} cm"}

    # ── VILLAGE ───────────────────────────────────────────────────────────
    m = re.search(r'(?:गांव|village|gram)\s+([A-Za-zऀ-ॿ]+)', tn, re.IGNORECASE)
    if m:
        fields["village"] = {"value": m.group(1).title(), "confidence": 0.85}

    # ── MUAC ─────────────────────────────────────────────────────────────
    m = re.search(r'(?:muac|बाहु)[^\d]*(\d+(?:\.\d+)?)', tn, re.IGNORECASE)
    if m:
        fields["muac_cm"] = {"value": float(m.group(1)), "confidence": 0.9}

    # ── DERIVED: nutrition status + WHO range ─────────────────────────────
    if "weight_kg" in fields and "age_months" in fields:
        status = nutrition_status(fields["weight_kg"]["value"],
                                  fields["age_months"]["value"])
        fields["nutrition_status"] = {
            "value": status["label"], "color": status["color"],
            "action": status["action"], "confidence": 1.0,
        }
    if "age_months" in fields:
        lo, hi = who_norm(fields["age_months"]["value"])
        fields["who_range"] = {"value": f"{lo}–{hi} kg", "confidence": 1.0}

    # ── DERIVED: BMI (when height + weight present, no age) ──────────────
    if "weight_kg" in fields and "height_cm" in fields:
        w = fields["weight_kg"]["value"]
        h = fields["height_cm"]["value"] / 100.0
        if h > 0.3:
            bmi = round(w / (h * h), 1)
            if bmi < 16:   cat, col = "Severely Underweight", "red"
            elif bmi < 18.5: cat, col = "Underweight", "orange"
            elif bmi < 25:   cat, col = "Normal / Healthy", "green"
            elif bmi < 30:   cat, col = "Overweight", "orange"
            else:            cat, col = "Obese", "red"
            fields["bmi"] = {"value": bmi, "color": col,
                             "action": cat, "confidence": 1.0}

    return fields


def _narrative(title, icon, severity, plain_text, problem, advice, steps, tracker, docs=None):
    """Build a standardised advisory narrative response dict."""
    pdf_sources = []
    if docs:
        for d in docs:
            if d.get("source") and d["source"] != "Poshan KB":
                pdf_sources.append({"title": d["title"], "source": d["source"],
                                    "page": d.get("page"), "snippet": d.get("snippet","")})
    return {
        "title": title, "icon": icon, "severity": severity,
        "plain_text": plain_text,
        "narrative": {"problem": problem, "advice": advice,
                      "steps": steps, "tracker": tracker},
        "pdf_sources": pdf_sources[:3],
    }


# ── RAG ANSWER ────────────────────────────────────────────────────────────────
def rag_answer(question: str, docs: list, fields: dict = None) -> dict:  # noqa: C901
    q = normalize(question)
    fields = fields or {}

    # ── BMI / FITNESS CHECK (height + weight given in same query) ─────────
    if "weight_kg" in fields and "height_cm" in fields and "bmi" in fields:
        w   = fields["weight_kg"]["value"]
        h   = fields["height_cm"]["value"]
        bmi = fields["bmi"]["value"]
        cat = fields["bmi"]["action"]
        col = fields["bmi"]["color"]
        name = fields.get("child_name", {}).get("value", "")
        name_str = f"{name}'s" if name else "The person's"
        sev_map = {"red": "critical", "orange": "warning", "green": "info"}
        sev = sev_map.get(col, "info")

        if bmi < 18.5:
            advice_detail = ("Increase caloric intake with nutrient-dense foods — whole grains, dal, eggs, dairy, "
                             "nuts, and healthy fats like ghee. Eat 5–6 smaller meals per day rather than 3 large ones. "
                             "Rule out underlying causes (TB, worm infestation, malabsorption) if weight gain is poor despite adequate diet.")
            steps = [f"Current BMI {bmi} is below the healthy range (18.5–24.9) → classified as {cat}",
                     "Increase daily calorie intake: add ghee/oil, nuts, eggs, paneer to meals",
                     "Eat 5–6 times a day — do not skip breakfast",
                     "Get a blood test: Haemoglobin, serum albumin, and worm load (stool test)",
                     "If no improvement in 4 weeks, refer to Medical Officer or district hospital",
                     "Record weight monthly; target 0.5–1 kg gain per month"]
        elif bmi < 25:
            advice_detail = ("This is a healthy BMI range. Maintain a balanced diet with adequate protein, fibre, and micronutrients. "
                             "30 minutes of physical activity daily is recommended. Continue regular health check-ups.")
            steps = [f"Current BMI {bmi} is in the NORMAL range (18.5–24.9) — no immediate action needed",
                     "Maintain balanced diet: grains, dal, vegetables, fruit, dairy, protein daily",
                     "30 minutes moderate physical activity daily (walking, cycling, yoga)",
                     "Annual health check-up including blood pressure and haemoglobin",
                     "Record weight every 3 months to monitor any drift"]
        elif bmi < 30:
            advice_detail = ("BMI in the overweight range increases risk of diabetes, hypertension, and heart disease. "
                             "Reduce refined carbohydrates (white rice, maida, sugar) and increase vegetables, pulses, and lean protein. "
                             "Aim for at least 45 minutes of brisk walking daily.")
            steps = [f"Current BMI {bmi} is in the OVERWEIGHT range (25–29.9) → action needed",
                     "Reduce refined carbs: white rice, maida, sugar, fried snacks",
                     "Increase fibre: whole grains, vegetables, dal, salads",
                     "45 min brisk walk or equivalent exercise daily",
                     "Check blood pressure and fasting blood sugar at PHC",
                     "Target 0.5 kg weight loss per week — do not crash diet",
                     "Follow-up weight check every 4 weeks"]
        else:
            advice_detail = ("Obesity significantly increases risk of type 2 diabetes, hypertension, stroke, and joint problems. "
                             "Medical evaluation is essential. Dietary changes must be combined with structured physical activity "
                             "under supervision. Refer to district hospital for dietitian counselling.")
            steps = [f"Current BMI {bmi} is in the OBESE range (≥30) → medical evaluation required",
                     "Refer to Medical Officer or district hospital for formal assessment",
                     "Strict diet: eliminate sugar, fried food, processed snacks entirely",
                     "Structured exercise plan: start with 20–30 min walking, gradually increase",
                     "Check for comorbidities: blood sugar (diabetes), BP (hypertension), lipids",
                     "Monthly weight monitoring; target 1–2 kg loss per month under medical guidance"]

        ht_note = fields["height_cm"].get("note", f"{h} cm")
        return _narrative(
            title=f"BMI Assessment — {cat}",
            icon="⚖️", severity=sev,
            plain_text=(f"{name_str} BMI is {bmi} (weight {w} kg, height {h} cm). "
                        f"Classification: {cat}. {advice_detail[:200]}"),
            problem=(f"{name_str} height is {ht_note} and weight is {w} kg. "
                     f"Calculated BMI = {bmi}. According to WHO standards, this falls in the "
                     f"'{cat}' category (BMI {'< 18.5' if bmi < 18.5 else '18.5–24.9' if bmi < 25 else '25–29.9' if bmi < 30 else '≥ 30'})."),
            advice=advice_detail,
            steps=steps,
            tracker=f"Height ({h} cm) · Weight ({w} kg) · BMI ({bmi}) · Category ({cat}) · Date",
            docs=docs,
        )

    # ── WEIGHT NORMS ──────────────────────────────────────────────────────
    if any(w in q for w in ["वजन","वज़न","weight","wajan","kilo","किलो","kitna","normal weight","कितना होना","कितना होता","वज़न होना","वजन होना","fit","healthy","body weight","bmi","body mass"]):
        age = word_to_age_months(question)
        if age is not None:
            # ── ADULT (> 18 years = > 216 months) — use BMI, not child WHO norms ──
            if age > 216:
                age_y = age // 12
                w_val = fields.get("weight_kg", {}).get("value") if fields else None
                if w_val:
                    # Build BMI at common adult heights (Indian population range)
                    _ht_list = [150, 155, 160, 165, 170, 175]
                    _bmi_rows = []
                    for _hcm in _ht_list:
                        _b = round(w_val / (_hcm / 100) ** 2, 1)
                        _cat = ("Obese" if _b >= 30 else
                                "Overweight" if _b >= 25 else
                                "Normal" if _b >= 18.5 else "Underweight")
                        _bmi_rows.append(f"  {_hcm} cm → BMI {_b} ({_cat})")
                    bmi_table = "\n".join(_bmi_rows)

                    # Ideal weight range at typical Indian adult heights
                    _ideal_rows = []
                    for _hcm in _ht_list:
                        _h = _hcm / 100
                        _ideal_rows.append(
                            f"  {_hcm} cm → ideal weight {round(18.5*_h*_h,1)}–{round(24.9*_h*_h,1)} kg"
                        )
                    ideal_table = "\n".join(_ideal_rows)

                    # Overall category at the mid-point height (165 cm)
                    bmi_165 = round(w_val / (1.65 ** 2), 1)
                    h_norm_lo = round((w_val / 24.9) ** 0.5 * 100)
                    h_norm_hi = round((w_val / 18.5) ** 0.5 * 100)
                    if bmi_165 < 18.5:    overall = "Underweight"
                    elif bmi_165 < 25:    overall = "Normal weight"
                    elif bmi_165 < 30:    overall = "Overweight"
                    else:                 overall = "Obese"
                    sev = "critical" if bmi_165 >= 30 else "warning" if bmi_165 >= 25 else "info"

                    return _narrative(
                        title=f"Adult Weight Assessment — {age_y}-Year-Old, {w_val} kg",
                        icon="⚖️", severity=sev,
                        plain_text=(f"At {w_val} kg (age {age_y} years), estimated BMI is {bmi_165} at 165 cm height "
                                    f"→ {overall}. "
                                    f"For a normal BMI (18.5–24.9) the person should weigh between "
                                    f"{round(18.5*(1.65**2),1)}–{round(24.9*(1.65**2),1)} kg (at 165 cm). "
                                    f"Provide exact height for precise calculation."),
                        problem=(f"You asked about weight fitness for a {age_y}-year-old adult weighing {w_val} kg. "
                                 f"At 165 cm height, BMI = {bmi_165} → {overall}. "
                                 f"WHO child growth charts do NOT apply to adults — BMI is the correct measure."),
                        advice=(f"BMI = weight(kg) / height(m)². At {w_val} kg, a normal BMI (18.5–24.9) "
                                f"requires height {h_norm_lo}–{h_norm_hi} cm — which is unusually tall. "
                                f"For most Indian adults (150–170 cm height), {w_val} kg is in the Overweight or Obese range. "
                                f"To reach a healthy weight at 165 cm, target: "
                                f"{round(18.5*(1.65**2),1)}–{round(24.9*(1.65**2),1)} kg. "
                                f"Weight loss of 0.5–1 kg/week through diet + exercise is safe and sustainable."),
                        steps=[f"BMI at {w_val} kg across heights:\n{bmi_table}",
                               f"Ideal weight ranges by height:\n{ideal_table}",
                               f"Provide exact height (cm or feet) for a precise BMI — say e.g. '160 cm' or '5 feet 4 inch'",
                               f"If Overweight/Obese: reduce refined carbs, sugar, fried food; 45 min brisk walk daily",
                               f"Check BP, fasting blood sugar, and cholesterol at PHC — obesity raises risk of all three",
                               f"Safe weight loss target: 0.5–1 kg per week; do not crash diet"],
                        tracker=f"Adult age ({age_y} yr) · Weight ({w_val} kg) · Height (cm) · BMI ({bmi_165} at 165 cm) · BP · Blood sugar · Date",
                        docs=docs,
                    )
                else:
                    return _narrative(
                        title=f"Adult Weight — BMI Assessment",
                        icon="⚖️", severity="info",
                        plain_text=(f"For a {age_y}-year-old adult, healthy weight depends on height. "
                                    f"Normal adult BMI is 18.5–24.9. "
                                    f"Please provide weight in kg and height in cm or feet for a BMI calculation."),
                        problem=f"You are asking about normal weight for a {age_y}-year-old adult. WHO child norms do not apply.",
                        advice=(f"BMI (Body Mass Index) = weight(kg) / height(m)² is the standard adult weight measure. "
                                f"Normal BMI: 18.5–24.9. Underweight: <18.5. Overweight: 25–29.9. Obese: ≥30. "
                                f"Please say or type the person's weight (kg) AND height (cm or feet) to get the BMI."),
                        steps=["Say weight in kg: e.g. '65 kg' or '65 kilo'",
                               "Say height in cm or feet: e.g. '160 cm' or '5 feet 4 inch'",
                               "BMI = weight(kg) / height(m)²",
                               "Normal BMI 18.5–24.9: maintain diet + 30 min daily exercise",
                               "Annual check-up: blood pressure, fasting blood sugar, cholesterol"],
                        tracker="Adult weight (kg) · Height (cm) · BMI · BP · Blood sugar · Date",
                        docs=docs,
                    )

            # ── CHILD (≤ 18 years) — use WHO child growth norms ──────────────
            lo, hi = who_norm(age)
            nearby = sorted([(k, v) for k, v in WHO.items() if abs(k-age) <= 12])
            ref_rows = "\n".join(f"  • {k} months: {v[0]}–{v[1]} kg" for k, v in nearby)
            return _narrative(
                title=f"Weight Norm — {age}-Month Child",
                icon="⚖️", severity="info",
                plain_text=(f"For a {age}-month-old child, normal weight is {lo}–{hi} kg. "
                            f"Below {lo} kg: screen for MAM or SAM and measure MUAC. "
                            f"Above {hi} kg: dietary counselling needed. Record weight monthly."),
                problem=(f"You are asking about the normal weight range for a child aged {age} months. "
                         f"As per WHO Child Growth Standards, the reference range for this age is {lo}–{hi} kg."),
                advice=(f"A child at {age} months should weigh between {lo} kg and {hi} kg to be considered normally nourished. "
                        f"If the child's weight falls below {lo} kg, assess for undernutrition by measuring MUAC — "
                        f"a reading below 11.5 cm indicates SAM and requires immediate NRC referral. "
                        f"If weight is above {hi} kg, counsel the mother on balanced diet and reduced high-fat or sugary foods."),
                steps=[f"Weigh the child and compare with the WHO range: {lo}–{hi} kg for {age} months",
                       f"Weight < {lo} kg → Measure MUAC immediately; if MUAC < 11.5 cm → refer to NRC",
                       f"Weight < {lo} kg and MUAC ≥ 11.5 cm → classify as MAM; provide RUTF and monthly follow-up",
                       f"Weight > {hi} kg → Counsel mother on reducing fried/sweet foods; increase vegetables and dal",
                       "Record weight on growth chart; alert supervisor if weight drops two months in a row",
                       f"Nearby WHO references (for counselling):\n{ref_rows}"],
                tracker=f"Child weight (kg) · Age ({age} months) · WHO range ({lo}–{hi} kg) · Nutrition status",
                docs=docs,
            )
        # no age given — return full table
        table = "\n".join(f"  {k}m → {v[0]}–{v[1]} kg" for k, v in sorted(WHO.items()))
        return _narrative(
            title="WHO Weight-for-Age Reference Table",
            icon="📊", severity="info",
            plain_text="WHO weight norms (children 0–5 years): 0m: 2.5-4.5 kg, 6m: 5.5-9.0 kg, 12m: 7.0-11.5 kg, 24m: 9.0-15.0 kg, 60m: 13.5-22.0 kg. For adults use BMI.",
            problem="You are asking about standard weight ranges for children under 6 years.",
            advice=("The WHO Child Growth Standards define normal weight-for-age ranges from birth to 60 months. "
                    "For adults (18+ years), use BMI = weight(kg) / height(m)² — normal range 18.5–24.9. "
                    "Monthly weighing of all registered children is mandatory under Poshan Tracker."),
            steps=[f"WHO Weight-for-Age Reference (children):\n{table}",
                   "Find the child's age row and compare their weight to the normal range",
                   "Below the lower limit → assess MUAC and classify MAM/SAM",
                   "Weigh every registered child (0–6 years) every month without exception",
                   "For adults: provide weight + height to calculate BMI"],
            tracker="Monthly weight (kg) · Age in months · Nutrition status · Date",
            docs=docs,
        )

    # ── SAM / MAM ─────────────────────────────────────────────────────────
    if any(w in q for w in ["sam","mam","कुपोषण","malnutrition","severe","moderate","underweight","कमज़ोर","कुपोषित","गंभीर","सैम","मैम","मुआक"]):
        return _narrative(
            title="Acute Malnutrition — SAM & MAM",
            icon="🏥", severity="critical",
            plain_text=("SAM: MUAC < 11.5 cm or Z-score < -3 SD — refer to NRC immediately. "
                        "MAM: MUAC 11.5–12.5 cm — provide RUTF and monthly follow-up. "
                        "Both require IFA, Vitamin A, and counselling on complementary feeding."),
            problem=("The child is being assessed for acute malnutrition. "
                     "Malnutrition in children under 5 significantly increases mortality risk and causes "
                     "irreversible stunting and cognitive impairment if not treated promptly."),
            advice=("SAM (Severe Acute Malnutrition) is diagnosed when MUAC is below 11.5 cm, "
                    "weight-for-height Z-score is below −3 SD, or bilateral pitting oedema is present — any one criterion is enough. "
                    "SAM requires immediate referral to an NRC (Nutrition Rehabilitation Centre) for 14-day inpatient therapeutic feeding. "
                    "MAM (Moderate Acute Malnutrition) — MUAC 11.5–12.5 cm — is managed at the Anganwadi level "
                    "with RUTF distribution, complementary feeding counselling, and monthly monitoring."),
            steps=["Measure MUAC on the left arm at the midpoint between shoulder and elbow",
                   "SAM (MUAC < 11.5 cm): Refer to NRC TODAY — inform parents and supervisor the same day",
                   "SAM: Also check for bilateral pitting oedema (press foot 3 sec — if depression forms = oedema = SAM)",
                   "MAM (MUAC 11.5–12.5 cm): Distribute RUTF from Anganwadi stock; monthly MUAC follow-up",
                   "Both: Give IFA syrup (20 mg biweekly) and Vitamin A supplement if due",
                   "Both: Counsel mother on complementary feeding diversity and hygiene",
                   "Both: Report to CDPO if no improvement after 2 consecutive monthly assessments"],
            tracker="MUAC (cm) · Status (SAM/MAM/Normal) · NRC referral date · RUTF given · IFA administered",
            docs=docs,
        )

    # ── DIARRHOEA / ORS / ZINC ─────────────────────────────────────────────
    if any(w in q for w in ["diarrhea","diarrhoea","loose","ors","zinc","दस्त","पतला","उल्टी दस्त","dehydration","ओआरएस","जिंक","पानी की कमी","लूज़","पेट"]):
        return _narrative(
            title="Diarrhoea Management — ORS & Zinc Protocol",
            icon="💧", severity="warning",
            plain_text=("Give ORS after each loose stool: 50–100 ml for under 2 years, 100–200 ml for older. "
                        "Give zinc 20 mg/day for 14 days. Continue feeding. "
                        "Refer if sunken eyes, unable to drink, blood in stool, or diarrhoea ≥ 14 days."),
            problem=("The child has diarrhoea. Diarrhoea is the 2nd leading cause of under-5 deaths in India — "
                     "the primary danger is dehydration and electrolyte loss, which can become fatal within hours in young children."),
            advice=("Treatment has two pillars: ORS to replace fluids lost, and zinc to reduce the duration and severity of the episode. "
                    "Give ORS after every loose stool using the low-osmolarity formulation (mix one sachet in 1 litre clean water). "
                    "Give zinc daily for the full 14-day course even if diarrhoea resolves sooner — it reduces future episodes too. "
                    "Do NOT stop breastfeeding or food — continued feeding speeds recovery and prevents malnutrition."),
            steps=["Mix ORS sachet in 1 litre clean water; give after EACH loose stool",
                   "Children < 2 years: 50–100 ml ORS per stool",
                   "Children 2–10 years: 100–200 ml ORS per stool",
                   "Give zinc: 10 mg/day (age 2–6 months) OR 20 mg/day (6 months–5 years) for 14 days",
                   "Dissolve zinc dispersible tablet in breast milk or ORS — do not give dry",
                   "Continue breastfeeding and food throughout — do NOT restrict diet",
                   "REFER TO PHC IMMEDIATELY: sunken eyes, unable to drink, blood in stool, or episode ≥ 14 days"],
            tracker="Diarrhoea episode date · Duration (days) · ORS packets given · Zinc dose and days · PHC referral Y/N",
            docs=docs,
        )

    # ── VACCINATION ────────────────────────────────────────────────────────
    if any(w in q for w in ["vaccine","वैक्सीन","टीका","bcg","बीसीजी","opv","dpt","measles","खसरा","polio","vaccination","immunis","टीकाकरण","hepatitis","pentavalent","rotavirus","pcv","टीके","कब देना","inject"]):
        return _narrative(
            title="Universal Immunisation Programme (UIP) Schedule",
            icon="💉", severity="info",
            plain_text=("BCG, OPV-0, Hep-B at birth. OPV/DPT/Rotavirus/PCV at 6, 10, 14 weeks. "
                        "Measles + Vitamin A at 9 months. DPT booster + MR + Vitamin A at 16–24 months."),
            problem=("You are asking about the national childhood vaccination schedule. "
                     "Vaccine-preventable diseases (measles, polio, diphtheria, TB) remain major causes of child mortality — "
                     "missed doses must be identified and followed up within 30 days."),
            advice=("The Universal Immunisation Programme (UIP) mandates a fixed schedule from birth to 5 years. "
                    "BCG, OPV and Hepatitis B are given at birth; primary series of OPV, Pentavalent, Rotavirus and PCV are given "
                    "at 6, 10 and 14 weeks; Measles/MR vaccine and first Vitamin A dose are given at 9 months; "
                    "boosters at 16–24 months complete the primary schedule. "
                    "AWW tracks each child's vaccination status on the MCP card and in Poshan Tracker."),
            steps=["At birth: BCG + OPV-0 + Hepatitis B",
                   "6 weeks: OPV-1 + Pentavalent-1 + Rotavirus-1 + PCV-1",
                   "10 weeks: OPV-2 + Pentavalent-2 + Rotavirus-2 + PCV-2",
                   "14 weeks: OPV-3 + Pentavalent-3 + Rotavirus-3 + PCV-3 + IPV-1",
                   "9 months: Measles/MR dose 1 + Vitamin A dose 1 (1 lakh IU)",
                   "16–24 months: DPT booster + OPV booster + MR dose 2 + Vitamin A dose 2 (2 lakh IU)",
                   "Check MCP card monthly; identify missed doses; follow up dropout children within 30 days",
                   "Vaccines must be stored at 2–8°C — check cold chain when collecting from PHC"],
            tracker="Vaccine name · Date given · Batch no. · Next due date · Mark 'Drop-Out' for missed doses",
            docs=docs,
        )

    # ── IFA / ANAEMIA ──────────────────────────────────────────────────────
    if any(w in q for w in ["ifa","iron","आयरन","folic","tablet","गोली","फोलिक","anaemia","anemia","खून","हीमोग्लोबिन","hemoglobin","wifs","weekly","आईएफए","खून की कमी","एनीमिया","एनिमिया","लोह"]):
        return _narrative(
            title="Anaemia Prevention — IFA Supplementation (Anemia Mukt Bharat)",
            icon="💊", severity="warning",
            plain_text=("Children 6–59 months: 20 mg iron biweekly (pink syrup). "
                        "Adolescents 10–19 years: 100 mg weekly Monday (blue tablet). "
                        "Pregnant women: 100 mg iron daily, minimum 180 days (red tablet). "
                        "Lactating mothers: same daily for 180 days postpartum."),
            problem=("The beneficiary requires Iron and Folic Acid supplementation to prevent anaemia. "
                     "Anaemia affects over 50% of Indian children and women, impairing growth, cognition, and maternal outcomes."),
            advice=("IFA supplementation is the primary intervention under Anemia Mukt Bharat. "
                    "Each age group has a specific colour-coded tablet, dose, and frequency. "
                    "Monday has been declared the National Anaemia Control Day — all IFA distribution and supervised ingestion "
                    "should happen on Mondays. Dark stool after IFA is normal and harmless — counsel beneficiaries not to stop. "
                    "Biannual deworming with Albendazole 400 mg (August and February) is given alongside IFA for children and adolescents."),
            steps=["Monday = IFA Day: distribute and supervise tablet/syrup intake in person",
                   "Children 6–59 months: 20 mg iron + 100 mcg FA as PINK syrup, biweekly (52 doses/year)",
                   "Children 5–10 years: 45 mg iron as PINK tablet, weekly (52 tablets/year)",
                   "Adolescents 10–19 yrs: 100 mg iron as BLUE tablet, weekly + deworming twice a year",
                   "Pregnant women: 100 mg iron as RED tablet, daily from 1st trimester (min 180 days)",
                   "Lactating mothers: RED tablet daily for 180 days postpartum",
                   "Give IFA after meals; counsel that dark stool is normal; avoid tea/coffee within 1 hour",
                   "Refer to PHC if Hb < 7 g/dL (severe anaemia requiring medical evaluation)"],
            tracker="IFA tablets/syrup given per beneficiary · Compliance (taken/not taken) · Hb value if tested · Deworming date",
            docs=docs,
        )

    # ── REGISTRATION ───────────────────────────────────────────────────────
    if any(w in q for w in ["register","registration","पंजीकरण","fields","form","aadhaar","आधार","enroll","beneficiary"]):
        return _narrative(
            title="Beneficiary Registration — Required Fields",
            icon="📋", severity="info",
            plain_text=("Mandatory fields: full name, Aadhaar (12 digits), DOB, gender, mother's name, "
                        "mobile, village, gram panchayat, block, district, caste, BPL status, bank account. "
                        "For children additionally: birth weight, birth order, delivery type."),
            problem=("You are asking about registering a new beneficiary in Poshan Tracker. "
                     "All beneficiaries — children 0–6 years, pregnant women, lactating mothers, and adolescent girls — "
                     "must be registered to receive entitled services and DBT cash transfers."),
            advice=("Registration is the entry point for all Poshan 2.0 services. "
                    "Aadhaar is the primary identifier for deduplication across the system; if Aadhaar is not available, "
                    "use the birth certificate or MCP card number as a temporary ID and link Aadhaar later. "
                    "A valid, active bank account is mandatory for PMMVY and SNP direct benefit transfers. "
                    "Duplicate entries are auto-flagged by the platform — do not re-register existing beneficiaries."),
            steps=["Open Poshan Tracker → Beneficiary Registration → Select category",
                   "Enter Identity: full name, Aadhaar (12 digits), date of birth, gender",
                   "Enter Family: mother's name, father's/husband's name, active mobile number",
                   "Enter Location: village, gram panchayat, block, district, state",
                   "Enter Socioeconomic: caste (Gen/OBC/SC/ST), religion, BPL status, bank A/C for DBT",
                   "For child registration additionally: birth weight, birth order, delivery type",
                   "If no Aadhaar: use birth certificate or MCP card number; link Aadhaar once obtained",
                   "For pregnant women: set LMP date and Expected Delivery Date (EDD)"],
            tracker="Complete beneficiary profile · Photo (if state mandated) · AWC tag · EDD for pregnant women",
            docs=docs,
        )

    # ── MUAC ───────────────────────────────────────────────────────────────
    if any(w in q for w in ["muac","arm","circumference","बाहु","mid-upper","mid upper","बाजू","भुजा","मुआक","कलाई"]):
        return _narrative(
            title="MUAC — Mid-Upper Arm Circumference",
            icon="📏", severity="warning",
            plain_text=("Normal: MUAC ≥ 12.5 cm. MAM: 11.5–12.5 cm → provide RUTF. "
                        "SAM: < 11.5 cm → refer to NRC immediately. "
                        "Measure left arm midpoint between shoulder and elbow using colour-coded MUAC tape."),
            problem=("You are asking about MUAC measurement and interpretation. "
                     "MUAC is the most reliable community-level tool for detecting acute malnutrition in children 6–59 months — "
                     "it is independent of age and does not require a scale, making it ideal for routine screening at the Anganwadi."),
            advice=("Use the colour-coded MUAC tape from the Poshan kit. "
                    "Green (≥12.5 cm) means the child is well-nourished — continue monthly monitoring. "
                    "Yellow (11.5–12.5 cm) indicates MAM — provide RUTF, counsel on feeding, schedule monthly follow-up. "
                    "Red (< 11.5 cm) indicates SAM — this is a medical emergency requiring immediate NRC referral. "
                    "For pregnant women, MUAC below 21 cm indicates Chronic Energy Deficiency (CED) requiring intensified nutrition support."),
            steps=["Child should stand upright (or sit); always measure the LEFT arm",
                   "Find midpoint between shoulder tip (acromion) and elbow tip (olecranon); mark with pen",
                   "Wrap MUAC tape snugly (not tight) at the marked midpoint; read to nearest 0.1 cm",
                   "Green ≥12.5 cm: Record; schedule next monthly monitoring; no immediate action needed",
                   "Yellow 11.5–12.4 cm (MAM): Provide RUTF; counsel on feeding; monthly follow-up",
                   "Red < 11.5 cm (SAM): Refer to NRC IMMEDIATELY; inform parents and supervisor same day",
                   "Pregnant women: MUAC < 21 cm → intensify THR distribution and nutrition counselling"],
            tracker="MUAC reading (cm) · Classification (Green/Yellow/Red) · Date · Action taken (RUTF/NRC referral)",
            docs=docs,
        )

    # ── BREASTFEEDING / IYCF ──────────────────────────────────────────────
    if any(w in q for w in ["breastfeed","breast","स्तनपान","nursing","colostrum","कोलोस्ट्रम","feed","dudh","दूध","मां का दूध","स्तन","स्तनपान","छाती का दूध","पिलाना"]):
        return _narrative(
            title="Infant and Young Child Feeding (IYCF)",
            icon="🤱", severity="info",
            plain_text=("Initiate breastfeeding within 1 hour of birth. Give colostrum — never discard. "
                        "Exclusive breastfeeding 0–6 months — no water or food. "
                        "Start complementary foods at exactly 6 months; continue breastfeeding to 2 years."),
            problem=("You are asking about correct infant feeding practices. "
                     "Common harmful practices in rural communities — discarding colostrum, giving water to newborns, "
                     "starting solid food before 6 months — significantly increase infection risk and stunting."),
            advice=("Colostrum — the thick yellow milk in the first 2–3 days — is called the 'first vaccine' because it "
                    "is packed with antibodies and nutrients. It must never be discarded. "
                    "Exclusive breastfeeding means breastmilk only for the first 6 months — not even water, as breastmilk is 88% water. "
                    "At exactly 6 months, start complementary foods while continuing breastfeeding. "
                    "Feed 2–3 meals/day at 6–8 months, 3–4 meals at 9–11 months, and 3–4 meals plus 2 snacks at 12–24 months."),
            steps=["Initiate breastfeeding within 60 minutes of birth — skin-to-skin contact helps",
                   "Give colostrum (first yellow milk): do NOT discard — it is the infant's first immunisation",
                   "0–6 months: breastmilk ONLY — no water, no other liquids, no solid food",
                   "6 months: start complementary foods alongside breastfeeding — not before, not after",
                   "Add 1 tsp ghee/oil per meal to increase energy density",
                   "Aim for minimum 5 of 8 food groups daily (grains, dal, dairy, egg/meat, veg, fruit, breastmilk)",
                   "Continue breastfeeding until 2 years and beyond",
                   "AWW: conduct cooking demonstration monthly at Anganwadi using local affordable ingredients"],
            tracker="Breastfeeding initiated Y/N · Exclusive BF at 6 months Y/N · Complementary feeding start date",
            docs=docs,
        )

    # ── HBNC / NEWBORN CARE ────────────────────────────────────────────────
    if any(w in q for w in ["newborn","hbnc","नवजात","lbw","low birth","kangaroo","kmc","postnatal","42 days","umbilical","cord","jaundice","पीलिया"]):
        return _narrative(
            title="Home-Based Newborn Care (HBNC) — 0 to 42 Days",
            icon="👶", severity="warning",
            plain_text=("6 HBNC visits for institutional delivery (days 3, 7, 14, 21, 28, 42). "
                        "7 visits for home delivery (Day 1 added). "
                        "Assess breastfeeding, temperature, cord, danger signs. "
                        "Refer immediately for: not feeding, breathing difficulty, fever, convulsions, jaundice below chest."),
            problem=("You are asking about newborn care in the first 42 days of life. "
                     "Over 50% of under-5 deaths occur in the first 28 days — the neonatal period is the highest-risk window. "
                     "HBNC visits by ASHA (coordinated by AWW) are the primary tool for detecting and responding to neonatal danger signs."),
            advice=("For an institutional delivery, ASHA makes 6 home visits: on days 3, 7, 14, 21, 28, and 42 after birth. "
                    "For a home delivery, an additional visit on Day 1 (within 24 hours) is mandatory — making 7 visits total. "
                    "At each visit, check breastfeeding, temperature, umbilical cord condition, eyes, and for any danger signs. "
                    "LBW (< 2.5 kg) and preterm babies need Kangaroo Mother Care — skin-to-skin contact with the mother for at least 8 hours/day."),
            steps=["Institutional delivery: schedule visits on Days 3, 7, 14, 21, 28, 42",
                   "Home delivery: add Day 1 visit within 24 hours of birth",
                   "Check at each visit: breastfeeding initiated? Temperature normal? Cord clean and dry?",
                   "LBW baby: ensure Kangaroo Mother Care ≥ 8 hrs/day; room temperature 28°C minimum",
                   "REFER IMMEDIATELY if: baby not feeding, fast/difficult breathing, fever >37.5°C, convulsions, "
                   "jaundice spreading below chest, pus from umbilicus or eyes, baby appears very small or blue",
                   "Ensure Day 42 immunisation: BCG + OPV-0 + Hepatitis B",
                   "Coordinate with local ASHA to ensure all visits happen on schedule"],
            tracker="HBNC visit day · Breastfeeding status · Birth weight · Danger sign detected Y/N · Referral Y/N",
            docs=docs,
        )

    # ── PMMVY ──────────────────────────────────────────────────────────────
    if any(w in q for w in ["pmmvy","maternity","matru vandana","मातृ वंदना","cash","benefit","rs 5000","rs 6000","incentive","scheme","yojana"]):
        return _narrative(
            title="PMMVY — Pradhan Mantri Matru Vandana Yojana",
            icon="💰", severity="info",
            plain_text=("First child: Rs 5,000 in 2 instalments via DBT. "
                        "Instalment 1: Rs 3,000 after early pregnancy registration + 1 ANC. "
                        "Instalment 2: Rs 2,000 after delivery + 14-week immunisation. "
                        "Second child (if girl): Rs 6,000 single instalment."),
            problem=("You are asking about the PMMVY maternity benefit scheme. "
                     "PMMVY compensates pregnant and lactating women from disadvantaged backgrounds for wage loss during pregnancy, "
                     "while incentivising three critical health behaviours: early ANC registration, institutional delivery, and child immunisation."),
            advice=("Eligible women are those aged 19 and above, pregnant with their first or second child (second only if the child is a girl), "
                    "and belonging to SC/ST, disabled (≥40%), BPL, PMJAY, e-Shram, MGNREGA, or family income below Rs 8 lakh. "
                    "The benefit is transferred directly to the beneficiary's bank account via DBT — the bank account and Aadhaar must be linked. "
                    "Women in regular Central or State Government employment are not eligible."),
            steps=["Identify eligible pregnant women at first contact — check caste, BPL, or income eligibility",
                   "Ensure early pregnancy registration in Poshan Tracker AND PMMVY-CAS portal",
                   "Documents needed: Aadhaar (mother + husband), MCP card, bank passbook copy",
                   "Instalment 1 (Rs 3,000): Apply after at least 1 ANC visit is completed",
                   "Instalment 2 (Rs 2,000): Apply after delivery + BCG/OPV/DPT/Hep-B vaccinations done",
                   "Second girl child (Rs 6,000): Register during pregnancy; apply after delivery + immunisation",
                   "Follow up if DBT credit not received within 90 days; escalate to CDPO"],
            tracker="PMMVY application date · Instalment 1 date and amount · Instalment 2 date and amount · Bank credit confirmed Y/N",
            docs=docs,
        )

    # ── COMPLEMENTARY FEEDING ──────────────────────────────────────────────
    if any(w in q for w in ["complementary","solid","food","anna","अन्नप्राशन","porridge","weaning","ठोस","6 month","khana","khichdi","diet","ragi","millet","खाना","खिचड़ी","ऊपरी आहार","अन्न","खाद्य","आहार","रागी"]):
        return _narrative(
            title="Complementary Feeding Guidelines (6–24 Months)",
            icon="🥗", severity="info",
            plain_text=("Start at exactly 6 months. 6–8 months: 2–3 meals, 2–3 tbsp each; mashed texture. "
                        "9–11 months: 3–4 meals + snacks, half bowl; lumpy. "
                        "12–24 months: 3–4 meals + 2 snacks, full bowl; family food. "
                        "Minimum 5 of 8 food groups daily."),
            problem=("You are asking about when and how to start complementary foods for infants. "
                     "Early introduction (before 6 months) increases infection risk; late introduction (after 6 months) "
                     "causes growth faltering and stunting. Inadequate food diversity leads to micronutrient deficiencies."),
            advice=("Start complementary feeding at exactly 6 completed months — not a day earlier, and not later. "
                    "Continue breastfeeding alongside all complementary foods until the child is 2 years old. "
                    "Gradually increase frequency, portion size, and texture as the child grows. "
                    "Energy density matters — add 1 teaspoon of ghee or oil to each meal, which adds 45 kcal. "
                    "Ragi (finger millet) and dark green leafy vegetables are excellent local options for iron and calcium."),
            steps=["6–8 months: 2–3 meals/day; 2–3 tablespoons per meal; mashed/pureed texture + breastfeeding",
                   "9–11 months: 3–4 meals + 1–2 snacks/day; ½ bowl (125 ml); soft lumpy texture + breastfeeding",
                   "12–24 months: 3–4 meals + 2 snacks/day; 1 full bowl (250 ml); family food + breastfeeding",
                   "Add 1 tsp ghee/oil to each meal for energy density",
                   "Target minimum 5 of 8 food groups daily: grains, dal/nuts, dairy, egg/meat, Vitamin A veg/fruit, other fruits, other veg, breastmilk",
                   "DO NOT give before 1 year: honey (botulism risk), excess salt/sugar, cow's milk as main drink",
                   "AWW: conduct monthly cooking demonstration using local affordable ingredients"],
            tracker="CF start date · Dietary diversity score · THR distributed (kg) · HCM attendance (children/day)",
            docs=docs,
        )

    # ── VITAMIN A ──────────────────────────────────────────────────────────
    if any(w in q for w in ["vitamin a","विटामिन","vitamin","blindness","night blind","रतौंधी","xerophthalmia","vas","विटामिन ए","आंखें","दृष्टि","रतौंधी"]):
        return _narrative(
            title="Vitamin A Supplementation Schedule",
            icon="🟡", severity="info",
            plain_text=("Dose 1: 1 lakh IU at 9 months with measles vaccine. "
                        "Doses 2–9: 2 lakh IU every 6 months from 1 to 5 years — total 9 doses. "
                        "Signs of deficiency: night blindness, Bitot spots. Never give before 6 months."),
            problem=("You are asking about Vitamin A supplementation. "
                     "Vitamin A deficiency causes preventable blindness and increases the severity of childhood infections "
                     "including measles, diarrhoea, and pneumonia. India has one of the highest Vitamin A deficiency burdens globally."),
            advice=("9 mega doses of Vitamin A are given from 9 months to 5 years, every 6 months. "
                    "The first dose (1 lakh IU) is given at 9 months alongside the Measles/MR vaccine. "
                    "All subsequent doses (2 lakh IU each) are given every 6 months during routine immunisation sessions and VHNDs. "
                    "Vitamin A must never be given before 6 months, and a minimum 4-week gap must be maintained between doses. "
                    "If a child shows signs of Vitamin A deficiency (night blindness, Bitot spots on eyes), give a therapeutic dose immediately and refer to the Medical Officer."),
            steps=["9 months: give 1,00,000 IU Vitamin A alongside Measles/MR vaccine",
                   "16–18 months: give 2,00,000 IU alongside DPT booster",
                   "Then: 2,00,000 IU every 6 months until 5 years of age (Doses 3–9)",
                   "Track dose number (1–9) in MCP card and Poshan Tracker vaccination module",
                   "If child shows night blindness or Bitot spots: give therapeutic dose immediately + refer to MO",
                   "Never give Vitamin A before 6 months; minimum 4-week gap between any two doses",
                   "Counsel mothers: include Vitamin A-rich foods daily — green leafy veg, carrot, mango, egg, liver"],
            tracker="Vitamin A dose number (1–9) · Date given · Batch number · Any adverse reaction",
            docs=docs,
        )

    # ── ANC ────────────────────────────────────────────────────────────────
    if any(w in q for w in ["anc","antenatal","prenatal","pregnancy","pregnant","गर्भ","गर्भावस्था","lmp","edd","trimester","bp","blood pressure","preeclampsia"]):
        return _narrative(
            title="Antenatal Care (ANC) Schedule",
            icon="🤰", severity="info",
            plain_text=("Minimum 4 ANC visits (target 8). ANC-1 before 12 weeks: register, Hb, BP, IFA, TT. "
                        "ANC-3 at 28–30 weeks: Hb and BP review. ANC-4 at 36 weeks: birth preparedness. "
                        "Refer immediately if BP ≥ 140/90 or Hb < 7 g/dL."),
            problem=("You are asking about antenatal care for a pregnant woman. "
                     "Early and regular ANC is the primary mechanism to detect pregnancy complications — "
                     "anaemia, pre-eclampsia, malnutrition — before they become life-threatening."),
            advice=("The national guideline recommends a minimum of 4 ANC visits, with a target of 8 visits. "
                    "The first visit must happen before 12 weeks for early risk assessment, Hb testing, and IFA tablet initiation. "
                    "At every visit: measure weight, blood pressure, and test haemoglobin. "
                    "Flag and refer immediately: Hb < 7 g/dL (severe anaemia) or BP ≥ 140/90 (pre-eclampsia risk). "
                    "TT vaccination, calcium supplementation, and birth preparedness counselling are mandatory."),
            steps=["ANC-1 (< 12 weeks): Register in Poshan Tracker + MCP card; check Hb, BP, blood group; start IFA + calcium; TT vaccine",
                   "ANC-2 (14–16 weeks): BP and weight check; verify IFA compliance; counsel on diet and rest",
                   "ANC-3 (28–30 weeks): Hb review (flag if < 11 g/dL); check fundal height and foetal heart rate; BP (flag if ≥ 140/90)",
                   "ANC-4 (36 weeks): Birth preparedness — identify delivery facility; review MCP card completeness; danger signs counselling",
                   "Ensure TT-1 + TT-2 vaccination (4 weeks apart) if not previously immunised",
                   "Refer to PHC/CHC immediately: severe anaemia (Hb < 7), high BP (≥ 140/90), reduced foetal movement, bleeding"],
            tracker="LMP · EDD · ANC visit dates · BP readings · Hb values · IFA tablets given · TT vaccination status · Delivery outcome",
            docs=docs,
        )

    # ── GENERIC FALLBACK — best RAG doc (manual + PDF) ────────────────────
    if docs:
        d = docs[0]
        pts = [l.strip() for l in d["full"].split("\n")
               if l.strip() and not l.startswith("#") and len(l.strip()) > 20][:6]
        plain = d.get("snippet", " ".join(pts))[:400].rstrip("…").strip()
        return _narrative(
            title=d["title"],
            icon="📖", severity="info",
            plain_text=plain,
            problem=plain,
            advice=plain,
            steps=pts if pts else ["Refer to your Poshan Tracker guidelines",
                                   "Contact your CDPO or supervisor for case-specific guidance"],
            tracker="Record all assessments, actions taken, and dates in Poshan Tracker",
            docs=docs,
        )

    return _narrative(
        title="Information Not Found",
        icon="❓", severity="info",
        plain_text="Please refer to your Poshan Tracker guidelines or supervisor for this query.",
        problem="Your question did not match any topic in the Poshan AI knowledge base.",
        advice=("Please rephrase your question or speak more clearly. "
                "This system covers: weight norms, SAM/MAM, vaccination, IFA/anaemia, MUAC, "
                "breastfeeding, diarrhoea, PMMVY, complementary feeding, HBNC, and Vitamin A. "
                "For specific case guidance, contact your CDPO or Medical Officer."),
        steps=["Try asking about a specific topic: 'What is the normal weight for a 12-month-old?'",
               "Or a clinical concern: 'What should I do if a child's MUAC is 11 cm?'",
               "Contact your CDPO or supervisor for urgent case-specific guidance"],
        tracker="—",
        docs=docs,
    )


# ── SARVAM TTS ────────────────────────────────────────────────────────────────
LANG_SPEAKER = {
    "hi":"meera","ta":"pavithra","te":"arvind","mr":"isha",
    "bn":"isha","gu":"isha","kn":"isha","ml":"isha","pa":"isha","or":"isha",
    "as":"isha",
}

def sarvam_tts(text: str, lang: str) -> str:
    if lang == "en":
        return None   # Sarvam TTS does not support English
    speaker   = LANG_SPEAKER.get(lang, "meera")
    lang_code = sarvam_lc(lang)
    try:
        r = requests.post(
            "https://api.sarvam.ai/text-to-speech",
            headers={"api-subscription-key": SARVAM_KEY,
                     "Content-Type": "application/json"},
            json={"inputs": [text[:500]], "target_language_code": lang_code,
                  "speaker": speaker, "model": "bulbul:v1",
                  "pace": 1.0, "loudness": 1.5},
            timeout=15,
        )
        if r.status_code == 200:
            audios = r.json().get("audios", [])
            return audios[0] if audios else None
    except Exception:
        pass
    return None


# ── SARVAM STT ────────────────────────────────────────────────────────────────
def sarvam_stt(wav_path: str, lang: str) -> tuple:
    lang_code = sarvam_lc(lang)
    t0 = time.time()
    with open(wav_path, "rb") as f:
        r = requests.post(
            "https://api.sarvam.ai/speech-to-text",
            headers={"api-subscription-key": SARVAM_KEY},
            files={"file": ("audio.wav", f, "audio/wav")},
            data={"language_code": lang_code, "model": "saaras:v3",
                  "with_timestamps": "false"},
            timeout=30,
        )
    r.raise_for_status()
    return r.json().get("transcript", ""), round(time.time() - t0, 2)


# ── PYDANTIC MODELS ───────────────────────────────────────────────────────────
class TranslateRequest(BaseModel):
    text:   str
    source: str = "en"
    target: str = "hi"

class TextQueryRequest(BaseModel):
    text: str
    lang: str = "hi"


# ── ROUTES ────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...), lang: str = "hi"):
    raw = await audio.read()
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as fin:
        fin.write(raw); fin_path = fin.name
    wav = fin_path.replace(".webm", ".wav")
    try:
        subprocess.run(["ffmpeg","-y","-i",fin_path,"-ar","16000","-ac","1",wav],
                       capture_output=True, timeout=15, check=True)
    except Exception as e:
        raise HTTPException(500, f"Audio conversion failed: {e}")

    try:
        transcript, stt_latency = await asyncio.get_event_loop().run_in_executor(
            None, sarvam_stt, wav, lang)
    except Exception as e:
        raise HTTPException(502, f"STT failed: {e}")
    finally:
        for p in [fin_path, wav]:
            try: os.unlink(p)
            except: pass

    query       = transcript or "child nutrition poshan tracker"
    rag_task    = asyncio.get_event_loop().run_in_executor(None, rag_search, query)
    fields_task = asyncio.get_event_loop().run_in_executor(None, extract_fields, transcript)
    rag_results, fields = await asyncio.gather(rag_task, fields_task)

    mode   = "question" if is_question(transcript) else "entry"
    answer = rag_answer(transcript, rag_results, fields) if mode == "question" else None

    # Build and translate the full advisory text (problem + advice + top 3 steps)
    translated_answer = None
    tts_audio         = None
    answer_en = None
    if mode == "question" and answer:
        # Use only plain_text for body translation (≤400 chars → reliable Sarvam API)
        body_en   = answer.get("plain_text", "")[:400]
        answer_en = body_en  # always keep English version for translate toggle

        if lang == "en":
            translated_answer = body_en
        else:
            translated_answer = await asyncio.get_event_loop().run_in_executor(
                None, sarvam_translate, body_en, "en", lang)
            # If Sarvam returns English unchanged (API failure), keep it — frontend handles it

        tts_src   = body_en
        tts_audio = await asyncio.get_event_loop().run_in_executor(
            None, sarvam_tts, tts_src, lang)

    entry = {"transcript": transcript, "lang": lang, "mode": mode,
             "fields": fields, "answer": answer, "stt_latency": stt_latency}
    HISTORY.appendleft(entry)

    return JSONResponse({
        "transcript":        transcript,
        "mode":              mode,
        "lang":              lang,
        "answer":            answer,
        "translated_answer": translated_answer,
        "answer_en":         answer_en,
        "tts_audio":         tts_audio,
        "fields":            fields,
        "rag":               rag_results,
        "stt_latency":       stt_latency,
    })


@app.post("/api/translate")
async def translate(req: TranslateRequest):
    result = await asyncio.get_event_loop().run_in_executor(
        None, sarvam_translate, req.text, req.source, req.target)
    return JSONResponse({"translated": result, "source": req.source, "target": req.target})


@app.post("/api/text")
async def text_query(req: TextQueryRequest):
    """Process a typed text query — same pipeline as transcribe but skips STT."""
    transcript = req.text.strip()
    lang       = req.lang
    if not transcript:
        raise HTTPException(400, "Empty query")

    rag_task    = asyncio.get_event_loop().run_in_executor(None, rag_search, transcript)
    fields_task = asyncio.get_event_loop().run_in_executor(None, extract_fields, transcript)
    rag_results, fields = await asyncio.gather(rag_task, fields_task)

    mode   = "question" if is_question(transcript) else "entry"
    answer = rag_answer(transcript, rag_results, fields) if mode == "question" else None

    answer_en         = None
    translated_answer = None
    tts_audio         = None
    if mode == "question" and answer:
        body_en   = answer.get("plain_text", "")[:400]
        answer_en = body_en
        if lang == "en":
            translated_answer = body_en
        else:
            translated_answer = await asyncio.get_event_loop().run_in_executor(
                None, sarvam_translate, body_en, "en", lang)
        tts_audio = await asyncio.get_event_loop().run_in_executor(
            None, sarvam_tts, body_en, lang)

    entry = {"transcript": transcript, "lang": lang, "mode": mode,
             "fields": fields, "answer": answer, "stt_latency": None}
    HISTORY.appendleft(entry)

    return JSONResponse({
        "transcript":        transcript,
        "mode":              mode,
        "lang":              lang,
        "answer":            answer,
        "translated_answer": translated_answer,
        "answer_en":         answer_en,
        "tts_audio":         tts_audio,
        "fields":            fields,
        "rag":               rag_results,
        "stt_latency":       None,
    })


@app.get("/api/health")
async def health():
    return JSONResponse({
        "status": "ok",
        "rag": RAG_OK,
        "docs": len(_texts) if RAG_OK else 0,
        "cache_size": len(_rag_cache),
    })


@app.get("/api/history")
async def get_history():
    return JSONResponse({"history": list(HISTORY)})


@app.get("/", response_class=HTMLResponse)
async def root():
    return _HTML


# ── FRONTEND ──────────────────────────────────────────────────────────────────
_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Poshan AI Assistant</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Noto+Sans+Devanagari:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root{
  --g9:#1a3a2a;--g8:#1e5c38;--g7:#25703f;--g5:#3aa05e;--g3:#8ecba5;--g1:#c8e6d0;--g0:#eef7f1;
  --red:#c62828;--red-lt:#fff0f0;--red-b:#b71c1c;--red-border:#fca5a5;
  --orange:#c05300;--orange-lt:#fff7ed;--orange-border:#fed7aa;
  --blue:#1046a0;--blue2:#1565c0;--blue-lt:#eff6ff;--blue-border:#bfdbfe;
  --teal:#00695c;--teal-lt:#f0fdfa;--teal-border:#99f6e4;
  --amber:#92400e;--amber-lt:#fffbeb;--amber-border:#fde68a;
  --ink:#0f172a;--ink2:#1e293b;--muted:#64748b;--muted2:#94a3b8;
  --border:#e2e8f0;--border2:#f1f5f9;--surface:#f8fafc;--white:#fff;
  --sidebar-w:68px;
  --sh1:0 1px 3px rgba(0,0,0,.06),0 1px 2px rgba(0,0,0,.04);
  --sh2:0 4px 12px rgba(0,0,0,.08),0 2px 4px rgba(0,0,0,.04);
  --sh3:0 8px 24px rgba(0,0,0,.10),0 4px 8px rgba(0,0,0,.06);
  --r-sm:8px;--r-md:12px;--r-lg:16px;--r-xl:20px;
}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Inter','Noto Sans Devanagari',system-ui,sans-serif;
  background:var(--surface);color:var(--ink);height:100dvh;overflow:hidden;display:flex}

/* ── SIDEBAR (desktop) ───────────────────────────────────────────────── */
.sidebar{width:var(--sidebar-w);background:var(--g9);display:flex;flex-direction:column;
  align-items:center;padding:14px 0;flex-shrink:0;z-index:10}
.sb-brand{width:40px;height:40px;background:var(--g7);border-radius:var(--r-md);
  display:flex;align-items:center;justify-content:center;font-size:20px;
  margin-bottom:24px;cursor:pointer;box-shadow:var(--sh1)}
.sb-nav{display:flex;flex-direction:column;gap:4px;flex:1}
.sb-item{width:48px;height:48px;border-radius:var(--r-md);display:flex;align-items:center;
  justify-content:center;font-size:20px;cursor:pointer;transition:all .15s;
  position:relative;color:rgba(255,255,255,.6)}
.sb-item:hover{background:rgba(255,255,255,.1);color:#fff}
.sb-item.active{background:rgba(255,255,255,.15);color:#fff}
.sb-item.active::before{content:'';position:absolute;left:0;top:50%;transform:translateY(-50%);
  width:3px;height:24px;background:var(--g3);border-radius:0 3px 3px 0}
.sb-item .sb-tip{position:absolute;left:58px;background:var(--ink2);color:#fff;
  font-size:11px;font-weight:600;padding:5px 10px;border-radius:var(--r-sm);
  white-space:nowrap;pointer-events:none;opacity:0;transition:opacity .15s;z-index:99;
  box-shadow:var(--sh2)}
.sb-item:hover .sb-tip{opacity:1}
.sb-profile{width:36px;height:36px;border-radius:50%;background:rgba(255,255,255,.12);
  display:flex;align-items:center;justify-content:center;font-size:17px;cursor:pointer;
  border:1.5px solid rgba(255,255,255,.18);margin-top:8px;transition:all .15s}
.sb-profile:hover{background:rgba(255,255,255,.2)}

/* ── MAIN ───────────────────────────────────────────────────────────── */
.main{flex:1;display:flex;flex-direction:column;overflow:hidden;min-width:0}

/* ── TOP BAR ────────────────────────────────────────────────────────── */
.topbar{height:58px;background:var(--white);border-bottom:1px solid var(--border);
  display:flex;align-items:center;padding:0 20px;gap:12px;flex-shrink:0;box-shadow:var(--sh1)}
.tb-brand{display:flex;align-items:center;gap:10px;flex-shrink:0}
.tb-logo{width:32px;height:32px;background:var(--g7);border-radius:8px;
  display:flex;align-items:center;justify-content:center;font-size:16px}
.tb-title{font-size:15px;font-weight:700;color:var(--ink)}
.tb-sub{font-size:11px;color:var(--muted);font-weight:400;display:none}
.tb-search{flex:1;max-width:360px;margin:0 auto;position:relative}
.tb-search input{width:100%;padding:8px 14px 8px 36px;border:1.5px solid var(--border);
  border-radius:24px;font-size:13px;background:var(--surface);outline:none;
  color:var(--ink);font-family:inherit;transition:all .15s}
.tb-search input:focus{border-color:var(--g5);background:#fff;box-shadow:0 0 0 3px rgba(58,160,94,.1)}
.tb-si{position:absolute;left:12px;top:50%;transform:translateY(-50%);
  font-size:14px;color:var(--muted2);pointer-events:none}
.tb-right{display:flex;align-items:center;gap:8px;margin-left:auto}
.rag-pill{display:flex;align-items:center;gap:5px;background:var(--g0);
  border:1px solid var(--g1);border-radius:20px;padding:4px 10px;flex-shrink:0}
.rag-dot{width:7px;height:7px;border-radius:50%;background:var(--g5);
  animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.5}}
.rag-label{font-size:11px;color:var(--g8);font-weight:600}
.tb-icon{width:38px;height:38px;border-radius:var(--r-md);display:flex;align-items:center;
  justify-content:center;font-size:17px;cursor:pointer;transition:background .15s;
  color:var(--muted);position:relative}
.tb-icon:hover{background:var(--surface)}
.tb-avatar{width:36px;height:36px;border-radius:50%;background:var(--g7);
  display:flex;align-items:center;justify-content:center;font-size:13px;
  color:#fff;font-weight:700;cursor:pointer;flex-shrink:0;
  box-shadow:0 0 0 2px var(--g1)}

/* ── CHAT FEED ──────────────────────────────────────────────────────── */
.chat-feed{flex:1;overflow-y:auto;padding:20px 20px 12px;
  display:flex;flex-direction:column;gap:12px;scroll-behavior:smooth}
.chat-feed::-webkit-scrollbar{width:4px}
.chat-feed::-webkit-scrollbar-thumb{background:var(--border);border-radius:4px}

/* Welcome */
.welcome{text-align:center;margin:auto;max-width:520px;padding:24px 16px}
.welcome-icon{font-size:52px;margin-bottom:12px;filter:drop-shadow(0 2px 4px rgba(0,0,0,.1))}
.welcome h2{font-size:22px;font-weight:800;color:var(--ink);margin-bottom:8px;letter-spacing:-.3px}
.welcome p{font-size:14px;color:var(--muted);line-height:1.7;margin-bottom:20px;max-width:380px;margin-inline:auto}
.welcome-features{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:20px}
.welcome-feat{background:var(--white);border:1px solid var(--border);border-radius:var(--r-md);
  padding:12px 8px;text-align:center;font-size:12px;color:var(--muted2)}
.welcome-feat .wf-icon{font-size:20px;margin-bottom:4px}
.welcome-feat .wf-lbl{font-weight:600;color:var(--ink2);font-size:11.5px}
.welcome-chips{display:flex;flex-wrap:wrap;gap:8px;justify-content:center}
.wchip{background:var(--white);border:1.5px solid var(--border);color:var(--ink2);
  font-size:12.5px;padding:8px 14px;border-radius:20px;cursor:pointer;font-weight:500;
  transition:all .15s;box-shadow:var(--sh1);text-align:left;line-height:1.4}
.wchip:hover{background:var(--g0);border-color:var(--g3);color:var(--g8);transform:translateY(-1px);box-shadow:var(--sh2)}

/* Messages — centered 900px column, no edge-hugging */
.msg-row{display:flex;gap:10px;align-items:flex-start;
  max-width:900px;width:100%;margin:0 auto}
.msg-row.user{flex-direction:row-reverse}
.msg-row.user .msg-body{align-items:flex-end}
.msg-avatar{width:34px;height:34px;border-radius:50%;display:flex;align-items:center;
  justify-content:center;font-size:15px;flex-shrink:0;margin-top:2px;box-shadow:var(--sh1)}
.msg-row.user .msg-avatar{background:var(--blue2);color:#fff}
.msg-row.ai .msg-avatar{background:var(--g8);color:#fff}
.msg-body{display:flex;flex-direction:column;gap:3px;min-width:0;width:100%}
.msg-meta{font-size:10px;color:var(--muted2);font-weight:500;
  display:flex;align-items:center;gap:5px;flex-wrap:wrap;margin-bottom:1px}
.meta-lang{background:var(--g0);color:var(--g8);padding:2px 7px;
  border-radius:5px;font-weight:700;font-size:9.5px;border:1px solid var(--g1);letter-spacing:.02em}
.meta-lat{color:var(--teal);font-weight:700;font-size:9.5px}

/* User bubble — constrained width, aligned right inside body */
.user-bubble{background:linear-gradient(135deg,var(--blue2) 0%,#1976d2 100%);
  color:#fff;border-radius:18px 4px 18px 18px;
  padding:12px 16px;font-size:14px;line-height:1.7;word-break:break-word;
  display:inline-flex;align-items:flex-start;gap:9px;box-shadow:var(--sh2);
  max-width:75%}
.vwave{font-size:16px;flex-shrink:0;opacity:.85;margin-top:1px}

/* ── ADVISORY CARD ──────────────────────────────────────────────────── */
.adv-card{background:var(--white);border-radius:var(--r-xl);
  border:1px solid var(--border);box-shadow:var(--sh2);
  overflow:hidden;transition:box-shadow .2s}
.adv-card:hover{box-shadow:var(--sh3)}
.adv-card.sev-critical{border-left:4px solid var(--red)}
.adv-card.sev-warning{border-left:4px solid var(--orange)}
.adv-card.sev-info{border-left:4px solid var(--g5)}

/* Card header */
.card-header{display:flex;align-items:flex-start;gap:12px;
  padding:16px 18px 0;background:var(--white)}
.card-icon-wrap{width:40px;height:40px;border-radius:var(--r-md);
  display:flex;align-items:center;justify-content:center;font-size:20px;flex-shrink:0}
.sev-critical .card-icon-wrap{background:var(--red-lt)}
.sev-warning .card-icon-wrap{background:var(--orange-lt)}
.sev-info .card-icon-wrap{background:var(--g0)}
.card-header-text{flex:1;min-width:0}
.card-title{font-size:15px;font-weight:800;color:var(--ink);line-height:1.3;letter-spacing:-.2px}
.card-badges{display:flex;align-items:center;gap:5px;margin-top:4px;flex-wrap:wrap}
.sev-badge{font-size:9.5px;font-weight:800;padding:2px 8px;border-radius:20px;
  text-transform:uppercase;letter-spacing:.5px}
.sev-badge.critical{background:var(--red-lt);color:var(--red-b);border:1px solid var(--red-border)}
.sev-badge.warning{background:var(--orange-lt);color:var(--orange);border:1px solid var(--orange-border)}
.sev-badge.info{background:var(--g0);color:var(--g8);border:1px solid var(--g1)}
.src-badge{font-size:9.5px;font-weight:600;padding:2px 8px;border-radius:20px;
  background:var(--blue-lt);color:var(--blue2);border:1px solid var(--blue-border)}

/* Quick answer */
.card-body{padding:12px 18px}
.quick-answer{font-size:14px;line-height:1.85;color:var(--ink2);word-break:break-word;
  font-feature-settings:"kern" 1}

/* Warning callout */
.warn-callout{margin:0 18px 12px;border-radius:var(--r-md);padding:12px 14px;
  display:flex;gap:10px;align-items:flex-start;border:1px solid}
.warn-callout.critical{background:var(--red-lt);border-color:var(--red-border)}
.warn-callout.warning{background:var(--amber-lt);border-color:var(--amber-border)}
.warn-icon{font-size:16px;flex-shrink:0;margin-top:1px}
.warn-text{font-size:12.5px;line-height:1.6;font-weight:500}
.critical .warn-text{color:var(--red-b)}
.warning .warn-text{color:var(--amber)}

/* Steps */
.steps-block{padding:0 18px 14px;border-top:1px solid var(--border2);margin-top:4px;padding-top:14px}
.steps-header{font-size:10.5px;font-weight:800;text-transform:uppercase;letter-spacing:.6px;
  color:var(--g8);margin-bottom:10px;display:flex;align-items:center;gap:6px}
.steps-list{list-style:none;display:flex;flex-direction:column;gap:8px}
.steps-list li{display:flex;gap:10px;font-size:13.5px;line-height:1.6;color:var(--ink2);
  padding:2px 0}
.step-num{min-width:22px;height:22px;border-radius:50%;background:var(--g0);
  color:var(--g7);font-size:10.5px;font-weight:800;display:flex;align-items:center;
  justify-content:center;flex-shrink:0;margin-top:1px;border:1.5px solid var(--g1)}
.steps-more{margin-top:8px}
.steps-toggle{background:none;border:1.5px solid var(--border);border-radius:20px;
  padding:4px 14px;font-size:12px;font-weight:600;color:var(--g8);cursor:pointer;
  display:flex;align-items:center;gap:5px;transition:all .15s}
.steps-toggle:hover{background:var(--g0);border-color:var(--g3)}

/* Tracker */
.tracker-block{margin:0 18px 14px;background:var(--teal-lt);border-radius:var(--r-md);
  padding:11px 14px;border:1px solid var(--teal-border)}
.tracker-lbl{font-size:10px;font-weight:800;text-transform:uppercase;letter-spacing:.5px;
  color:var(--teal);margin-bottom:5px;display:flex;align-items:center;gap:5px}
.tracker-txt{font-size:12.5px;color:var(--teal);line-height:1.65;font-weight:500}

/* PDF sources */
.src-row{padding:0 18px 14px;display:flex;flex-wrap:wrap;gap:5px}
.src-chip{font-size:10.5px;background:var(--surface);border:1px solid var(--border);
  color:var(--muted);padding:4px 10px;border-radius:20px;display:flex;align-items:center;gap:4px}
.src-chip::before{content:'📄';font-size:10px}

/* Response actions */
.resp-actions{display:flex;gap:6px;flex-wrap:wrap;padding:12px 18px 14px;
  border-top:1px solid var(--border2)}
.ra{display:flex;align-items:center;gap:5px;font-size:12px;font-weight:600;
  padding:7px 13px;border-radius:20px;border:1.5px solid var(--border);
  background:var(--white);color:var(--ink2);cursor:pointer;transition:all .15s;
  min-height:34px;white-space:nowrap}
.ra:hover{background:var(--surface);border-color:var(--g3);color:var(--g8)}
.ra.primary{background:var(--g8);color:#fff;border-color:var(--g8)}
.ra.primary:hover{background:var(--g7)}
.ra:disabled{opacity:.4;cursor:not-allowed}
.ra.active{background:var(--g0);border-color:var(--g5);color:var(--g8)}

/* Follow-up chips */
.followup-row{padding:4px 18px 14px;display:flex;flex-wrap:wrap;gap:6px}
.fu-chip{background:var(--surface);border:1.5px solid var(--border);color:var(--ink2);
  font-size:12px;padding:6px 13px;border-radius:20px;cursor:pointer;font-weight:500;
  transition:all .15s;display:flex;align-items:center;gap:5px}
.fu-chip::before{content:'↗';font-size:11px;color:var(--g7);font-weight:700}
.fu-chip:hover{background:var(--g0);border-color:var(--g3);color:var(--g8)}

/* Voice extraction card — sits between user message and advisory */
.extraction-card{background:var(--g0);border:1px solid var(--g1);
  border-radius:var(--r-lg);border-left:3px solid var(--g5);padding:12px 16px}
.ext-header{font-size:10px;font-weight:800;color:var(--g8);text-transform:uppercase;
  letter-spacing:.55px;margin-bottom:10px;display:flex;align-items:center;gap:5px}
.ext-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(100px,1fr));gap:10px}
.ext-field{display:flex;flex-direction:column;gap:2px}
.ext-lbl{font-size:9px;text-transform:uppercase;color:var(--muted);font-weight:700;letter-spacing:.4px}
.ext-val{font-size:15px;font-weight:800;color:var(--ink2);line-height:1.2}
.ext-note{font-size:10px;color:var(--muted);margin-top:1px;line-height:1.3}
.ext-field.ext-green .ext-val{color:var(--g7)}
.ext-field.ext-orange .ext-val{color:var(--orange)}
.ext-field.ext-red .ext-val{color:var(--red-b)}
.ext-field.ext-teal .ext-val{color:var(--teal)}

/* Key observations block inside advisory card */
.obs-block{padding:0 18px 12px;border-top:1px solid var(--border2);padding-top:12px}
.obs-lbl{font-size:10px;font-weight:800;text-transform:uppercase;letter-spacing:.55px;
  color:var(--muted);margin-bottom:8px;display:flex;align-items:center;gap:5px}
.obs-list{list-style:none;display:flex;flex-direction:column;gap:5px;padding:0;margin:0}
.obs-list li{font-size:13px;color:var(--ink2);line-height:1.6;
  padding-left:14px;position:relative}
.obs-list li::before{content:"•";position:absolute;left:0;color:var(--g5);font-weight:700}

/* Fields grid */
.fgrid{display:grid;grid-template-columns:1fr 1fr;gap:8px;padding:14px 18px}
.fi{background:var(--surface);border:1px solid var(--border);border-radius:var(--r-md);padding:12px 14px}
.fi-lbl{font-size:10px;text-transform:uppercase;color:var(--muted);font-weight:700;letter-spacing:.5px}
.fi-val{font-size:22px;font-weight:800;color:var(--blue2);margin-top:2px;line-height:1.2}
.fi-note{font-size:11px;color:var(--muted);margin-top:3px;font-weight:500}
.fi.green{background:var(--g0);border-color:var(--g1)}.fi.green .fi-val{color:var(--g7)}
.fi.orange{background:var(--orange-lt);border-color:var(--orange-border)}.fi.orange .fi-val{color:var(--orange)}
.fi.red{background:var(--red-lt);border-color:var(--red-border)}.fi.red .fi-val{color:var(--red-b)}
.fi.teal{background:var(--teal-lt);border-color:var(--teal-border)}.fi.teal .fi-val{color:var(--teal)}

/* Typing indicator */
.typing-card{background:var(--white);border:1px solid var(--border);
  border-radius:4px 20px 20px 20px;padding:14px 18px;
  display:flex;align-items:center;gap:10px;width:fit-content;box-shadow:var(--sh1)}
.typing-lbl{font-size:12px;color:var(--muted);font-weight:500}
.typing-dots{display:flex;gap:4px}
.typing-dots span{width:7px;height:7px;border-radius:50%;background:var(--g3);
  animation:tdot 1.3s infinite}
.typing-dots span:nth-child(2){animation-delay:.15s;background:var(--g5)}
.typing-dots span:nth-child(3){animation-delay:.3s;background:var(--g7)}
@keyframes tdot{0%,80%,100%{transform:scale(.7);opacity:.4}40%{transform:scale(1.1);opacity:1}}

/* ── INPUT PANEL ────────────────────────────────────────────────────── */
.input-panel{background:var(--white);border-top:1px solid var(--border);
  padding:10px 16px 14px;flex-shrink:0}
.lang-strip{display:flex;gap:4px;overflow-x:auto;margin-bottom:10px;
  scrollbar-width:none;-webkit-overflow-scrolling:touch;padding-bottom:2px}
.lang-strip::-webkit-scrollbar{display:none}
.lbtn{padding:5px 12px;border-radius:20px;border:1.5px solid var(--border);
  background:transparent;color:var(--muted);font-size:12px;cursor:pointer;
  transition:all .13s;line-height:1.3;white-space:nowrap;flex-shrink:0;
  display:flex;flex-direction:column;align-items:center;min-height:34px}
.lbtn:hover{background:var(--g0);border-color:var(--g3);color:var(--g8)}
.lbtn.on{background:var(--g8);border-color:var(--g8);font-weight:700;color:#fff}
.lbtn .lsub{font-size:8px;color:inherit;opacity:.7;line-height:1;margin-top:1px}

/* Unified input bar */
.input-bar{display:flex;align-items:center;gap:8px;
  background:var(--surface);border:1.5px solid var(--border);
  border-radius:26px;padding:6px 8px 6px 16px;transition:all .2s;
  box-shadow:var(--sh1)}
.input-bar:focus-within{border-color:var(--g5);background:var(--white);
  box-shadow:0 0 0 3px rgba(58,160,94,.1)}
.input-bar.recording{border-color:var(--red-b);background:var(--red-lt);
  box-shadow:0 0 0 3px rgba(198,40,40,.1);animation:rec-border 1.5s infinite}
@keyframes rec-border{0%,100%{box-shadow:0 0 0 3px rgba(198,40,40,.1)}50%{box-shadow:0 0 0 5px rgba(198,40,40,.18)}}
.bar-text{flex:1;border:none;background:transparent;font-size:14px;font-family:inherit;
  color:var(--ink);outline:none;padding:4px 0;min-width:0}
.bar-text::placeholder{color:var(--muted2)}
.bar-status{font-size:13px;color:var(--muted);flex:1;user-select:none;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap;padding:4px 0}
.bar-status.rec{color:var(--red-b);font-weight:600;animation:blink-txt 1s infinite}
.bar-status.ok{color:var(--g7);font-weight:600}
@keyframes blink-txt{0%,100%{opacity:1}50%{opacity:.4}}
.bar-divider{width:1px;height:24px;background:var(--border);flex-shrink:0}
.bar-btn{width:36px;height:36px;border-radius:50%;border:none;background:transparent;
  color:var(--muted);cursor:pointer;font-size:17px;display:flex;align-items:center;
  justify-content:center;transition:all .15s;flex-shrink:0}
.bar-btn:hover{background:var(--border);color:var(--ink)}
.mic-btn{width:46px;height:46px;border-radius:50%;border:none;
  background:var(--g9);color:#fff;font-size:21px;cursor:pointer;
  display:flex;align-items:center;justify-content:center;flex-shrink:0;
  transition:all .18s;box-shadow:0 3px 10px rgba(0,0,0,.25);position:relative}
.mic-btn:hover{transform:scale(1.07);background:var(--g7)}
.mic-btn.rec{background:var(--red-b)}
.mic-btn.rec::before{content:'';position:absolute;inset:-4px;border-radius:50%;
  border:2px solid var(--red-b);animation:ring 1.2s infinite}
@keyframes ring{0%{opacity:.8;transform:scale(1)}100%{opacity:0;transform:scale(1.5)}}
.send-btn{width:40px;height:40px;border-radius:50%;background:var(--g7);border:none;
  color:#fff;display:flex;align-items:center;justify-content:center;
  cursor:pointer;flex-shrink:0;transition:all .15s;box-shadow:var(--sh1)}
.send-btn:hover{background:var(--g8);transform:scale(1.05)}
canvas#wave{height:32px;flex-shrink:0;display:none;width:80px}

/* Toast */
.toast{position:fixed;bottom:76px;left:50%;transform:translateX(-50%);
  background:rgba(15,23,42,.92);color:#fff;padding:9px 18px;border-radius:20px;
  font-size:13px;font-weight:600;z-index:999;opacity:0;transition:opacity .25s;
  pointer-events:none;backdrop-filter:blur(4px);white-space:nowrap;max-width:90vw;
  text-align:center}
.toast.show{opacity:1}

/* Notification dropdown */
.notif-badge{position:absolute;top:-4px;right:-4px;background:var(--red-b);color:#fff;
  font-size:9px;font-weight:800;min-width:16px;height:16px;border-radius:8px;
  display:flex!important;align-items:center;justify-content:center;padding:0 3px;pointer-events:none}
.notif-dropdown{position:fixed;top:62px;right:12px;width:310px;max-height:420px;
  background:var(--white);border:1px solid var(--border);border-radius:var(--r-lg);
  box-shadow:var(--sh3);z-index:200;overflow:hidden;display:flex;flex-direction:column}
.notif-header{display:flex;align-items:center;justify-content:space-between;
  padding:13px 16px;border-bottom:1px solid var(--border);background:var(--surface)}
.notif-title{font-size:13px;font-weight:700;color:var(--ink)}
.notif-clear{font-size:11px;color:var(--g7);cursor:pointer;font-weight:600}
.notif-clear:hover{text-decoration:underline}
.notif-list{overflow-y:auto;flex:1}
.notif-empty{color:var(--muted);font-size:13px;text-align:center;padding:28px 16px}
.notif-item{padding:11px 16px;border-bottom:1px solid var(--border2);cursor:default}
.notif-item:last-child{border-bottom:none}
.notif-item-title{font-size:12.5px;font-weight:700;color:var(--ink)}
.notif-item-body{font-size:11.5px;color:var(--muted);margin-top:3px;line-height:1.4}
.notif-item.unread{background:var(--g0)}

/* Tab panels */
.tab-panel{flex:1;overflow-y:auto;display:flex;flex-direction:column;background:var(--surface)}
.panel-header{display:flex;align-items:center;gap:10px;padding:16px 20px;
  border-bottom:1px solid var(--border);background:var(--white);
  position:sticky;top:0;z-index:2;box-shadow:var(--sh1)}
.panel-title{font-size:15px;font-weight:700;color:var(--g8);flex:1}
.panel-btn{background:var(--white);border:1.5px solid var(--border);color:var(--ink2);
  font-size:12px;font-weight:600;padding:6px 14px;border-radius:20px;cursor:pointer;
  transition:all .15s}
.panel-btn:hover{background:var(--g0);border-color:var(--g3);color:var(--g8)}
.panel-body{padding:18px 20px;flex:1;display:flex;flex-direction:column;gap:12px}
.panel-empty{color:var(--muted);font-size:13.5px;text-align:center;margin-top:48px;line-height:1.7}
.log-entry{background:var(--white);border:1px solid var(--border);
  border-radius:var(--r-md);padding:13px 16px;box-shadow:var(--sh1)}
.log-time{font-size:11px;color:var(--muted2);font-weight:600;margin-bottom:5px;
  display:flex;align-items:center;gap:6px}
.log-text{font-size:13px;color:var(--ink);line-height:1.55}
.stat-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}
.stat-card{background:var(--white);border:1px solid var(--border);border-radius:var(--r-md);
  padding:18px 12px;text-align:center;box-shadow:var(--sh1)}
.stat-num{font-size:30px;font-weight:800;color:var(--g7);line-height:1}
.stat-lbl{font-size:11px;color:var(--muted);margin-top:4px;font-weight:500}
.set-section{background:var(--white);border:1px solid var(--border);
  border-radius:var(--r-md);padding:16px 18px;box-shadow:var(--sh1)}
.set-label{font-size:10.5px;font-weight:700;text-transform:uppercase;color:var(--muted);
  letter-spacing:.06em;margin-bottom:12px}
.api-status-box{font-size:13px;line-height:1.8;color:var(--ink2)}
.api-row{display:flex;align-items:center;gap:8px;margin-bottom:4px}
.api-dot{width:8px;height:8px;border-radius:50%;background:var(--g5)}
.api-dot.off{background:#ef4444}
.about-box{font-size:13px;line-height:1.75;color:var(--ink2)}
.lang-stat{display:flex;flex-direction:column;gap:6px}
.lang-bar-row{display:flex;align-items:center;gap:8px;font-size:12px;color:var(--ink2)}
.lang-bar-bg{flex:1;height:6px;background:var(--g1);border-radius:3px;overflow:hidden}
.lang-bar-fill{height:100%;background:var(--g5);border-radius:3px;transition:width .5s}

/* Mobile bottom nav */
.bottom-nav{display:none;position:fixed;bottom:0;left:0;right:0;
  background:var(--g9);height:58px;z-index:50;
  justify-content:space-around;align-items:center;
  border-top:1px solid rgba(255,255,255,.08);
  box-shadow:0 -4px 16px rgba(0,0,0,.15)}
.bn-item{display:flex;flex-direction:column;align-items:center;gap:3px;
  cursor:pointer;padding:6px 12px;border-radius:var(--r-md);
  color:rgba(255,255,255,.5);transition:all .15s;min-width:52px}
.bn-item:hover,.bn-item.active{color:#fff}
.bn-item.active{background:rgba(255,255,255,.1)}
.bn-icon{font-size:20px;line-height:1}
.bn-label{font-size:9.5px;font-weight:600;letter-spacing:.02em}

/* ── RESPONSIVE ──────────────────────────────────────────────────────── */
@media(min-width:769px){
  .tb-sub{display:inline}
  .welcome-features{grid-template-columns:repeat(3,1fr)}
}
@media(max-width:768px){
  .sidebar{display:none}
  .bottom-nav{display:flex}
  .main{padding-bottom:58px}
  .chat-feed{padding:14px 12px 10px}
  .input-panel{padding:8px 12px 10px}
  .msg-row{max-width:100%}
  .tb-search{display:none}
  .rag-pill{display:none}
  .fgrid{grid-template-columns:1fr}
  .stat-grid{grid-template-columns:repeat(2,1fr)}
  .welcome-features{grid-template-columns:repeat(2,1fr)}
  .toast{bottom:70px}
}
@media(max-width:480px){
  .card-header{padding:14px 14px 0}
  .card-body{padding:10px 14px}
  .steps-block,.tracker-block,.src-row,.resp-actions,.followup-row{padding-left:14px;padding-right:14px}
  .tracker-block{margin-left:14px;margin-right:14px}
  .src-row{padding-left:14px}
  .card-title{font-size:14px}
  .quick-answer{font-size:13.5px}
  .steps-list li{font-size:13px}
  .ra{font-size:11.5px;padding:6px 11px}
  .msg-avatar{width:30px;height:30px;font-size:13px}
  .welcome h2{font-size:19px}
  .tb-title{font-size:14px}
  .stat-grid{grid-template-columns:1fr 1fr}
  .welcome-features{grid-template-columns:repeat(2,1fr)}
}
</style>
</head>
<body>

<aside class="sidebar">
  <div class="sb-brand">🌾</div>
  <nav class="sb-nav">
    <div class="sb-item active" onclick="switchTab(this,'chat')">💬<span class="sb-tip">Chat</span></div>
    <div class="sb-item" onclick="switchTab(this,'log')">🕐<span class="sb-tip">Session Log</span></div>
    <div class="sb-item" onclick="switchTab(this,'analytics')">📊<span class="sb-tip">Analytics</span></div>
    <div class="sb-item" onclick="switchTab(this,'settings')">⚙️<span class="sb-tip">Settings</span></div>
  </nav>
  <div class="sb-profile" onclick="openSettings()" title="Profile & Settings">👤</div>
</aside>

<div class="main">
  <div class="topbar">
    <div class="tb-brand">
      <div class="tb-logo">🌾</div>
      <div><div class="tb-title">Poshan AI Assistant</div><div class="tb-sub">MoWCD · Poshan Tracker 2.0</div></div>
    </div>
    <div class="tb-search">
      <span class="tb-si">🔍</span>
      <input type="text" placeholder="Search guidelines…">
    </div>
    <div class="tb-right">
      <div class="rag-pill"><span class="rag-dot"></span><span class="rag-label">RAG Active</span></div>
      <div class="tb-icon" id="bellBtn" onclick="toggleNotif(event)" title="Notifications">🔔
        <span class="notif-badge" id="notifBadge" style="display:none">0</span>
      </div>
      <div class="tb-avatar" onclick="openSettings()" title="Settings">A</div>
    </div>
  </div>

  <div class="chat-feed" id="chatFeed">
    <div class="welcome" id="welcomeState">
      <div class="welcome-icon">🌾</div>
      <h2>Poshan AI Advisory System</h2>
      <p>Ask a clinical question in your language and receive structured advisory guidance based on official MoWCD, NHM and ICDS guidelines.</p>
      <div class="welcome-features">
        <div class="welcome-feat"><div class="wf-icon">🎙️</div><div class="wf-lbl">Voice Input</div>12 languages</div>
        <div class="welcome-feat"><div class="wf-icon">📋</div><div class="wf-lbl">WHO Norms</div>Instant advice</div>
        <div class="welcome-feat"><div class="wf-icon">📚</div><div class="wf-lbl">43 PDF guides</div>Official sources</div>
      </div>
      <div class="welcome-chips" id="welcomeChips"></div>
    </div>
  </div>

  <!-- Session Log Panel -->
  <div class="tab-panel" id="panelLog" style="display:none">
    <div class="panel-header">
      <span class="panel-title">🕐 Session Log</span>
      <button class="panel-btn" onclick="sessionLog.length=0;renderLogPanel()">Clear</button>
      <button class="panel-btn" onclick="exportLog()">⬇ Export</button>
    </div>
    <div class="panel-body" id="logBody">
      <div class="panel-empty">No entries yet — click <b>+ Add to Log</b> in any advisory to save it here.</div>
    </div>
  </div>

  <!-- Analytics Panel -->
  <div class="tab-panel" id="panelAnalytics" style="display:none">
    <div class="panel-header"><span class="panel-title">📊 Session Analytics</span></div>
    <div class="panel-body" id="analyticsBody"></div>
  </div>

  <!-- Settings Panel -->
  <div class="tab-panel" id="panelSettings" style="display:none">
    <div class="panel-header"><span class="panel-title">⚙️ Settings</span></div>
    <div class="panel-body">
      <div class="set-section">
        <div class="set-label">Default Language</div>
        <div class="lang-strip" id="settingsLangStrip" style="flex-wrap:wrap;gap:6px"></div>
      </div>
      <div class="set-section">
        <div class="set-label">API Status</div>
        <div id="apiStatusBox" class="api-status-box"></div>
      </div>
      <div class="set-section">
        <div class="set-label">About</div>
        <div class="about-box">
          <b>Poshan AI Assistant</b> — POC v3.0<br>
          MoWCD · Poshan Tracker 2.0<br><br>
          STT: Sarvam Saaras v3 · TTS: Bulbul v1 · Translate: Mayura v1<br>
          RAG: 22 manual KB docs + 5,759 PDF chunks (43 official MoWCD/NHM/ICDS documents)<br><br>
          <span style="color:var(--muted);font-size:11px">For authorized use by Anganwadi Workers and supervisors only.</span>
        </div>
      </div>
    </div>
  </div>

  <div class="input-panel" id="inputPanel">
    <div class="lang-strip" id="langStrip"></div>
    <div class="input-bar" id="inputBar">
      <input type="text" id="textInput" class="bar-text"
        placeholder="Type your question… (or tap mic)"
        onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();sendText();}">
      <span class="bar-status" id="inputStatus" style="display:none">Tap mic to speak</span>
      <canvas id="wave"></canvas>
      <button class="bar-btn" onclick="playLastTTS()" title="Listen again">🔊</button>
      <button class="bar-btn" onclick="clearChat()" title="Clear">🗑</button>
      <div class="bar-divider"></div>
      <button class="send-btn" onclick="sendText()" title="Send">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
      </button>
      <button class="mic-btn" id="micBtn" onclick="toggleRec()" title="Voice input">🎙️</button>
    </div>
  </div>
</div>

<!-- Mobile bottom navigation -->
<nav class="bottom-nav" id="bottomNav">
  <div class="bn-item active" onclick="switchBnTab(this,'chat')"><div class="bn-icon">💬</div><div class="bn-label">Chat</div></div>
  <div class="bn-item" onclick="switchBnTab(this,'log')"><div class="bn-icon">🕐</div><div class="bn-label">Log</div></div>
  <div class="bn-item" onclick="switchBnTab(this,'analytics')"><div class="bn-icon">📊</div><div class="bn-label">Analytics</div></div>
  <div class="bn-item" onclick="switchBnTab(this,'settings')"><div class="bn-icon">⚙️</div><div class="bn-label">Settings</div></div>
</nav>

<div class="toast" id="toast"></div>

<!-- Notification dropdown -->
<div class="notif-dropdown" id="notifDropdown" style="display:none">
  <div class="notif-header">
    <span class="notif-title">Notifications</span>
    <span class="notif-clear" onclick="clearNotifs()">Clear all</span>
  </div>
  <div class="notif-list" id="notifList">
    <div class="notif-empty">No notifications yet</div>
  </div>
</div>

<script>
const LANGS=[
  {c:"hi",l:"Hindi",   s:"हिं"},{c:"en",l:"English", s:"En"},
  {c:"ta",l:"Tamil",   s:"த"}, {c:"te",l:"Telugu",  s:"తె"},
  {c:"mr",l:"Marathi", s:"मर"},{c:"bn",l:"Bengali", s:"বাং"},
  {c:"gu",l:"Gujarati",s:"ગુ"},{c:"kn",l:"Kannada", s:"ಕ"},
  {c:"ml",l:"Malayalam",s:"മ"},{c:"pa",l:"Punjabi",  s:"ਪੰ"},
  {c:"or",l:"Odia",    s:"ଓ"}, {c:"as",l:"Assamese",s:"অস"},
];
const LMAP=Object.fromEntries(LANGS.map(l=>[l.c,l.l]));
const SAMPLES={
  hi:["14 mahine ke bachche ka sahi wajan kitna hona chahiye?","SAM kya hota hai?","BCG vaccine kab dena chahiye?","MUAC kaise measure karte hain?"],
  en:["Normal weight for 14 month old?","What is SAM and MAM?","When to give BCG vaccine?","How to measure MUAC?"],
  ta:["14 மாத குழந்தையின் சரியான எடை என்ன?","SAM என்றால் என்ன?"],
  te:["14 நెலல పిల్లవాని బరువు?","SAM అంటే ఏమిటి?"],
  mr:["14 महिन्याच्या मुलाचे सामान्य वजन किती?","SAM म्हणजे काय?"],
};

let lang="hi",isRec=false,recSec=0,timerInt=null;
let recorder,chunks=[],audioCtx,analyser,waveBuf,animFrame;
let lastTTS=null,lastFields=null,msgCount=0;

// Language strip
const ls=document.getElementById("langStrip");
LANGS.forEach(({c,l,s})=>{
  const b=document.createElement("button");
  b.className="lbtn"+(c==="hi"?" on":"");b.title=l;
  b.innerHTML=`${s}<span class="lsub">${l}</span>`;
  b.onclick=()=>{lang=c;document.querySelectorAll(".lbtn").forEach(x=>x.classList.remove("on"));b.classList.add("on");buildWelcomeChips();};
  ls.appendChild(b);
});

function buildWelcomeChips(){
  const el=document.getElementById("welcomeChips");if(!el)return;
  const exs=SAMPLES[lang]||SAMPLES.hi;
  el.innerHTML=exs.map(e=>`<span class="wchip" onclick="sendFollowUp('${e.replace(/'/g,"&#39;")}')">${e.length>52?e.slice(0,52)+"…":e}</span>`).join("");
}
buildWelcomeChips();

// ── Analytics tracking ────────────────────────────────────────────────
const sessionStats={queries:0,entries:0,langs:{},topics:{}};

function switchTab(el,tab){
  document.querySelectorAll(".sb-item").forEach(x=>x.classList.remove("active"));
  el.classList.add("active");
  const chatFeed=document.getElementById("chatFeed");
  const inputPanel=document.getElementById("inputPanel");
  const panels=["panelLog","panelAnalytics","panelSettings"];
  // show chat
  if(tab==="chat"){
    chatFeed.style.display="";inputPanel.style.display="";
    panels.forEach(p=>document.getElementById(p).style.display="none");
    return;
  }
  // hide chat feed + input for other tabs
  chatFeed.style.display="none";inputPanel.style.display="none";
  panels.forEach(p=>document.getElementById(p).style.display="none");
  if(tab==="log"){document.getElementById("panelLog").style.display="";renderLogPanel();}
  if(tab==="analytics"){document.getElementById("panelAnalytics").style.display="";renderAnalytics();}
  if(tab==="settings"){document.getElementById("panelSettings").style.display="";renderSettings();}
}

function renderLogPanel(){
  const body=document.getElementById("logBody");
  if(!sessionLog.length){
    body.innerHTML=`<div class="panel-empty">No entries yet — click <b>+ Add to Log</b> in any advisory to save it here.</div>`;
    return;
  }
  body.innerHTML=sessionLog.map((e,i)=>`
    <div class="log-entry">
      <div class="log-time">#${i+1} &nbsp;·&nbsp; ${e.time} &nbsp;·&nbsp; ${e.lang||""}</div>
      <div class="log-text">${esc(e.text)}</div>
    </div>`).join("");
}

function exportLog(){
  if(!sessionLog.length){toast("No log entries to export",2000);return;}
  const a=document.createElement("a");
  a.href=URL.createObjectURL(new Blob([JSON.stringify(sessionLog,null,2)],{type:"application/json"}));
  a.download="poshan_session_log.json";a.click();
  toast("Session log downloaded",2000);
}

function renderAnalytics(){
  const body=document.getElementById("analyticsBody");
  const total=sessionStats.queries+sessionStats.entries;
  const langEntries=Object.entries(sessionStats.langs).sort((a,b)=>b[1]-a[1]);
  const topicEntries=Object.entries(sessionStats.topics).sort((a,b)=>b[1]-a[1]).slice(0,5);
  const maxLang=langEntries[0]?.[1]||1;
  body.innerHTML=`
    <div class="stat-grid">
      <div class="stat-card"><div class="stat-num">${total}</div><div class="stat-lbl">Total Interactions</div></div>
      <div class="stat-card"><div class="stat-num">${sessionStats.queries}</div><div class="stat-lbl">Q&amp;A Queries</div></div>
      <div class="stat-card"><div class="stat-num">${sessionStats.entries}</div><div class="stat-lbl">Data Entries</div></div>
    </div>
    <div class="set-section">
      <div class="set-label">Language Usage</div>
      ${langEntries.length?`<div class="lang-stat">${langEntries.map(([l,n])=>`
        <div class="lang-bar-row">
          <span style="min-width:70px">${LMAP[l]||l}</span>
          <div class="lang-bar-bg"><div class="lang-bar-fill" style="width:${Math.round(n/maxLang*100)}%"></div></div>
          <span>${n}</span></div>`).join("")}</div>`:`<div style="color:var(--muted);font-size:13px">No data yet</div>`}
    </div>
    ${topicEntries.length?`<div class="set-section">
      <div class="set-label">Top Topics Asked</div>
      <div style="display:flex;flex-direction:column;gap:6px">${topicEntries.map(([t,n])=>`
        <div style="display:flex;justify-content:space-between;font-size:13px;padding:4px 0;border-bottom:1px solid var(--border)">
          <span>${esc(t)}</span><span style="color:var(--g7);font-weight:600">${n}</span></div>`).join("")}
      </div></div>`:""}`;
}

function renderSettings(){
  // populate settings lang strip if empty
  const sl=document.getElementById("settingsLangStrip");
  if(!sl.children.length){
    LANGS.forEach(lx=>{
      const b=document.createElement("button");
      b.className="lbtn"+(lx.c===lang?" on":"");
      b.textContent=lx.l;
      b.onclick=()=>{
        lang=lx.c;
        document.querySelectorAll(".lbtn").forEach(x=>x.classList.remove("on"));
        b.classList.add("on");
        // sync main lang strip too
        document.querySelectorAll("#langStrip .lbtn").forEach(x=>{
          x.classList.toggle("on",x.dataset&&x.dataset.lang===lang);
        });
        toast("Default language set to "+lx.l,2000);
      };
      sl.appendChild(b);
    });
  }
  // API status
  document.getElementById("apiStatusBox").innerHTML=`
    <div class="api-row"><span class="api-dot"></span>Sarvam STT (Saaras v3) — Connected</div>
    <div class="api-row"><span class="api-dot"></span>Sarvam TTS (Bulbul v1) — Connected</div>
    <div class="api-row"><span class="api-dot"></span>Sarvam Translate (Mayura v1) — Connected</div>
    <div class="api-row"><span class="api-dot"></span>FAISS RAG — 22 Manual + 5,759 PDF chunks loaded</div>`;
}

// Waveform
function startWave(stream){
  audioCtx=new(window.AudioContext||window.webkitAudioContext)();
  analyser=audioCtx.createAnalyser();analyser.fftSize=512;
  audioCtx.createMediaStreamSource(stream).connect(analyser);
  waveBuf=new Uint8Array(analyser.frequencyBinCount);
  const cv=document.getElementById("wave");cv.style.display="block";drawWave();
}
function drawWave(){
  const cv=document.getElementById("wave"),ctx=cv.getContext("2d");
  cv.width=cv.offsetWidth;cv.height=32;analyser.getByteTimeDomainData(waveBuf);
  ctx.clearRect(0,0,cv.width,cv.height);ctx.beginPath();
  ctx.strokeStyle="rgba(142,203,165,.9)";ctx.lineWidth=1.5;
  const sl=cv.width/waveBuf.length;
  waveBuf.forEach((v,i)=>{const x=i*sl,y=(v/128)*cv.height/2;i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);});
  ctx.stroke();animFrame=requestAnimationFrame(drawWave);
}
function stopWave(){
  cancelAnimationFrame(animFrame);
  const cv=document.getElementById("wave");cv.getContext("2d").clearRect(0,0,cv.width,cv.height);cv.style.display="none";
  if(audioCtx){audioCtx.close().catch(()=>{});audioCtx=null;}
}

// Recording
async function toggleRec(){isRec?stopRec():await startRec();}
async function startRec(){
  try{
    const stream=await navigator.mediaDevices.getUserMedia({audio:true});
    startWave(stream);chunks=[];
    recorder=new MediaRecorder(stream,{mimeType:"audio/webm"});
    recorder.ondataavailable=e=>e.data.size>0&&chunks.push(e.data);
    recorder.onstop=sendAudio;recorder.start(100);isRec=true;recSec=0;
    document.getElementById("inputBar").classList.add("recording");
    document.getElementById("textInput").style.display="none";
    document.getElementById("micBtn").classList.add("rec");
    document.getElementById("micBtn").textContent="⏹️";
    const st=document.getElementById("inputStatus");
    st.style.display="";st.textContent="Recording… speak now";st.className="bar-status rec";
    timerInt=setInterval(()=>{recSec++;document.getElementById("inputStatus").textContent=`Recording… ${recSec}s`;},1000);
  }catch(e){toast("Microphone access denied",3000);}
}
function stopRec(){
  clearInterval(timerInt);
  if(recorder&&recorder.state!=="inactive"){recorder.stop();recorder.stream.getTracks().forEach(t=>t.stop());}
  stopWave();isRec=false;
  document.getElementById("inputBar").classList.remove("recording");
  document.getElementById("micBtn").classList.remove("rec");
  document.getElementById("micBtn").textContent="🎙️";
  const st=document.getElementById("inputStatus");
  st.style.display="none";
  document.getElementById("textInput").style.display="";
}

// Send typed text
async function sendText(){
  const inp=document.getElementById("textInput");
  const txt=inp.value.trim();
  if(!txt)return;
  inp.value="";inp.disabled=true;
  const tid=addTyping();
  try{
    const res=await fetch(`/api/text`,{
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({text:txt,lang:lang})
    });
    if(!res.ok)throw new Error(await res.text());
    const d=await res.json();
    removeTyping(tid);render(d);
  }catch(e){
    removeTyping(tid);addErrMsg(e.message);
    toast("Error: "+e.message.slice(0,80),3000);
  }finally{inp.disabled=false;inp.focus();}
}

// Send audio
async function sendAudio(){
  const blob=new Blob(chunks,{type:"audio/webm"});
  const fd=new FormData();fd.append("audio",blob,"rec.webm");
  const tid=addTyping();
  try{
    const res=await fetch(`/api/transcribe?lang=${lang}`,{method:"POST",body:fd});
    if(!res.ok)throw new Error(await res.text());
    const d=await res.json();
    removeTyping(tid);render(d);
    const st=document.getElementById("inputStatus");
    st.textContent="Done — tap mic to continue";st.className="input-status ok";
  }catch(e){
    removeTyping(tid);addErrMsg(e.message);
    document.getElementById("inputStatus").textContent="Error — try again";
    document.getElementById("inputStatus").className="input-status";
  }
}

// Chat helpers
function scrollBottom(){const f=document.getElementById("chatFeed");f.scrollTop=f.scrollHeight;}
function hideWelcome(){const w=document.getElementById("welcomeState");if(w)w.style.display="none";}
function addTyping(){
  hideWelcome();const id="tp_"+(++msgCount);
  const r=document.createElement("div");r.className="msg-row ai";r.id=id;
  r.innerHTML=`<div class="msg-avatar">🤖</div><div class="msg-body"><div class="typing-card"><span class="typing-lbl">Analyzing…</span><div class="typing-dots"><span></span><span></span><span></span></div></div></div>`;
  document.getElementById("chatFeed").appendChild(r);scrollBottom();return id;
}
function removeTyping(id){const e=document.getElementById(id);if(e)e.remove();}
function addErrMsg(msg){
  const r=document.createElement("div");r.className="msg-row ai";
  r.innerHTML=`<div class="msg-avatar">⚠️</div><div class="msg-body"><div class="adv-card sev-critical" style="padding:14px 16px"><div style="color:var(--red-b);font-size:13px;font-weight:600">Error: ${esc(msg)}</div></div></div>`;
  document.getElementById("chatFeed").appendChild(r);scrollBottom();
}

// Main render
function render(d){
  hideWelcome();
  const ln=LMAP[d.lang]||d.lang;
  // track analytics
  if(d.mode==="question"){sessionStats.queries++;}else{sessionStats.entries++;}
  sessionStats.langs[d.lang]=(sessionStats.langs[d.lang]||0)+1;
  if(d.answer&&d.answer.title){
    const t=d.answer.title.replace(/\s*—.*$/,"").trim();
    sessionStats.topics[t]=(sessionStats.topics[t]||0)+1;
  }
  // User bubble — 🎙️ for voice, ✏️ for typed
  const isVoice=d.stt_latency!=null;
  const ur=document.createElement("div");ur.className="msg-row user";
  ur.innerHTML=`<div class="msg-avatar">👤</div><div class="msg-body">
    <div class="msg-meta"><span class="meta-lang">${ln}</span>${d.stt_latency?`<span class="meta-lat">${d.stt_latency}s</span>`:""}</div>
    <div class="user-bubble"><span class="vwave">${isVoice?"🎙️":"✏️"}</span><span>${esc(d.transcript||"(no speech)")}</span></div></div>`;
  document.getElementById("chatFeed").appendChild(ur);
  if(d.mode==="question"){
    // Show extraction summary between user query and advisory when fields were found
    const fc=Object.keys(d.fields||{}).length;
    if(fc>0) addExtractionSummary(d);
    addAiMsg(d,ln);
    if(d.answer)pushNotif(d.answer.title||"Advisory",
      (d.translated_answer||d.answer.plain_text||"").slice(0,80));
  } else {
    addExtractionSummary(d);
    const fc=Object.keys(d.fields||{}).length;
    if(fc)pushNotif("Data Entry Recorded",`${fc} field${fc>1?"s":""} extracted`);
  }
  scrollBottom();
}

const FOLLOWUP={
  "BMI":["What foods help gain weight?","Normal weight for this age?","When to refer to NRC?"],
  "Weight Norm":["How to measure MUAC?","What is SAM and MAM?","Foods for weight gain?"],
  "Acute Malnutrition":["How to refer to NRC?","What is RUTF?","When does MAM become SAM?"],
  "Diarrhoea":["How to prepare ORS?","When to refer to PHC?","What foods during diarrhoea?"],
  "Universal Immunisation":["Vaccines at 9 months?","How to store vaccines?","What is cold chain?"],
  "Anaemia":["Foods that increase iron?","When to refer for severe anaemia?","What is deworming?"],
  "MUAC":["SAM vs MAM difference?","When to give RUTF?","How to measure MUAC correctly?"],
  "Infant and Young Child":["When to start solid food?","What is exclusive breastfeeding?","Foods at 6 months?"],
  "Complementary Feeding":["Best foods at 6 months?","How much food per meal?","Can I give water before 6 months?"],
  "Vitamin A":["When is next Vitamin A dose?","Foods rich in Vitamin A?","Signs of Vitamin A deficiency?"],
  "Antenatal":["When is the first ANC visit?","TT vaccine schedule?","Hb level that needs referral?"],
  "PMMVY":["How to apply for PMMVY?","Documents needed for PMMVY?","When is the second instalment?"],
  "Home-Based Newborn":["What to check at HBNC visit?","When to refer a newborn?","What is Kangaroo Mother Care?"],
  "Beneficiary Registration":["What Aadhaar details are needed?","How to register a pregnant woman?","What if Aadhaar is not available?"],
};

function addAiMsg(d,ln){
  const ans=d.answer||{},nav=ans.narrative||{},sev=ans.severity||"info";
  const steps=nav.steps||[],sources=ans.pdf_sources||[];
  lastTTS=d.tts_audio;
  const id="m_"+(++msgCount);

  // Steps with expand/collapse for >3 steps
  let stHtml="";
  if(steps.length){
    const vis=steps.slice(0,3),rest=steps.slice(3);
    const visList=vis.map((s,i)=>`<li><span class="step-num">${i+1}</span><span>${esc(s)}</span></li>`).join("");
    const restPart=rest.length?`<div id="sr_${id}" style="display:none"><ul class="steps-list" style="margin-top:8px">${
      rest.map((s,i)=>`<li><span class="step-num">${i+4}</span><span>${esc(s)}</span></li>`).join("")
    }</ul></div><div class="steps-more"><button class="steps-toggle" id="st_${id}" onclick="toggleSteps('${id}')">▼ ${rest.length} more steps</button></div>`:"";
    stHtml=`<div class="steps-block"><div class="steps-header">✅ Action Steps</div><ul class="steps-list">${visList}</ul>${restPart}</div>`;
  }

  // Tracker
  let trHtml="";
  if(nav.tracker&&nav.tracker!=="—"){
    trHtml=`<div class="tracker-block"><div class="tracker-lbl">📱 Record in Poshan Tracker</div><div class="tracker-txt">${esc(nav.tracker)}</div></div>`;
  }

  // PDF sources
  let srHtml="";
  if(sources.length){
    srHtml=`<div class="src-row">${sources.map(s=>`<span class="src-chip">${esc(s.title||s.source)}${s.page?" p."+s.page:""}</span>`).join("")}</div>`;
  }

  // Warning callout for critical/warning severity
  let warnHtml="";
  if(sev==="critical"&&nav.problem){
    warnHtml=`<div class="warn-callout critical"><span class="warn-icon">🚨</span><span class="warn-text">${esc(nav.problem.slice(0,200))}</span></div>`;
  } else if(sev==="warning"&&nav.problem){
    warnHtml=`<div class="warn-callout warning"><span class="warn-icon">⚠️</span><span class="warn-text">${esc(nav.problem.slice(0,200))}</span></div>`;
  }

  // Follow-up chips
  const fkey=Object.keys(FOLLOWUP).find(k=>ans.title&&ans.title.startsWith(k));
  const fuHtml=fkey?`<div class="followup-row">${FOLLOWUP[fkey].map(c=>`<span class="fu-chip" onclick="sendFollowUp('${c.replace(/'/g,"&#39;")}')">${esc(c)}</span>`).join("")}</div>`:"";

  // Key observations — first 2-3 sentences from nav.advice
  let obsHtml="";
  if(nav.advice){
    const sents=(nav.advice||"").split(/(?<=[.!?।])\s+/).map(s=>s.trim()).filter(s=>s.length>25).slice(0,3);
    if(sents.length>0){
      obsHtml=`<div class="obs-block"><div class="obs-lbl">📋 Key observations</div>
        <ul class="obs-list">${sents.map(s=>`<li>${esc(s)}</li>`).join("")}</ul></div>`;
    }
  }

  // Source count badge
  const rc=d.rag&&d.rag.length?d.rag.length:0;
  const srcBadge=rc?`<span class="src-badge">📖 ${rc} source${rc>1?"s":""}</span>`:"";

  const r=document.createElement("div");r.className="msg-row ai";r.id=id;
  r.innerHTML=`<div class="msg-avatar">🤖</div><div class="msg-body">
    <div class="msg-meta">Poshan AI <span class="meta-lang">${ln}</span>${d.stt_latency?` <span class="meta-lat">${d.stt_latency}s</span>`:""}</div>
    <div class="adv-card sev-${sev}">
      <div class="card-header">
        <div class="card-icon-wrap">${ans.icon||"📋"}</div>
        <div class="card-header-text">
          <div class="card-title">${esc(ans.title||"Advisory")}</div>
          <div class="card-badges"><span class="sev-badge ${sev}">${sev.toUpperCase()}</span>${srcBadge}</div>
        </div>
      </div>
      <div class="card-body"><div class="quick-answer" id="ab_${id}">${esc(d.translated_answer||"")}</div></div>
      ${obsHtml}${warnHtml}${stHtml}${trHtml}${srHtml}
      <div class="resp-actions">
        <button class="ra primary" onclick="playTTS('${id}')">🔊 Listen</button>
        ${d.lang!=="en"?`<button class="ra" id="xb_${id}" onclick="xlat('${id}','${d.lang}')">↔ English</button>`:`<button class="ra" onclick="speakText('${id}')">🔊 Speak</button>`}
        <button class="ra" onclick="addToLog('${id}')">+ Log</button>
        <button class="ra" onclick="fb(this,1)">👍</button>
        <button class="ra" onclick="fb(this,-1)">👎</button>
      </div>
      ${fuHtml}
    </div></div>`;
  if(d.tts_audio) r.dataset.tts=d.tts_audio;
  r.dataset.orig=d.translated_answer||"";
  r.dataset.olang=d.lang||"en";
  r.dataset.eng=d.answer_en||d.translated_answer||"";
  r.dataset.advText=[nav.problem,nav.advice,(nav.steps||[]).slice(0,3).join(". ")].filter(Boolean).join(" ");
  document.getElementById("chatFeed").appendChild(r);
}

const FMETA={
  child_name:      {l:"Child Name",      c:""},
  weight_kg:       {l:"Weight (kg)",     c:""},
  age_months:      {l:"Age (months)",    c:""},
  height_cm:       {l:"Height (cm)",     c:""},
  village:         {l:"Village",         c:""},
  muac_cm:         {l:"MUAC (cm)",       c:""},
  who_range:       {l:"WHO Norm Range",  c:"teal"},
  nutrition_status:{l:"Nutrition Status",c:null},
  bmi:             {l:"BMI",             c:null},
};

function renderFields(fields){
  const keys=Object.keys(fields);
  if(!keys.length) return `<div style="color:var(--muted);font-size:13px;margin-top:8px">No fields extracted — mention child name, weight, age, or height.</div>`;
  return `<div class="fgrid">${keys.map(k=>{
    const m=FMETA[k]||{l:k,c:""};
    const f=fields[k];
    const c=f.color||m.c||"";
    const val=Array.isArray(f.value)?f.value.join(", "):f.value;
    return `<div class="fi ${c}">
      <div class="fi-lbl">${m.l}</div>
      <div class="fi-val">${val}</div>
      ${f.action?`<div class="fi-note">${f.action}</div>`:""}
      ${f.note?`<div class="fi-note">${f.note}</div>`:""}
    </div>`;
  }).join("")}</div>`;
}

function addExtractionSummary(d){
  const fields=d.fields||{};
  const primaryOrder=["child_name","age_months","weight_kg","height_cm","muac_cm","village"];
  const derivedOrder=["who_range","nutrition_status","bmi"];
  const primary=primaryOrder.filter(k=>fields[k]);
  const derived=derivedOrder.filter(k=>fields[k]);
  if(!primary.length&&!derived.length)return;
  lastFields=fields;

  const labelMap={child_name:"Name",weight_kg:"Weight",age_months:"Age",
    height_cm:"Height",muac_cm:"MUAC",village:"Village",
    who_range:"WHO Range",nutrition_status:"Status",bmi:"BMI"};
  const unitSuffix={weight_kg:" kg",height_cm:" cm",muac_cm:" cm"};

  const renderExtField=(k)=>{
    const f=fields[k];
    let val=String(f.value);
    if(k==="age_months")val=val+" months";
    else if(unitSuffix[k])val=val+unitSuffix[k];
    const col=f.color||"";
    const colCls=col?` ext-${col==="green"?"green":col==="red"?"red":col==="orange"?"orange":"teal"}`:"";
    const note=f.action?`<span class="ext-note">${esc(f.action)}</span>`:"";
    return `<div class="ext-field${colCls}"><span class="ext-lbl">${labelMap[k]||k}</span><span class="ext-val">${esc(val)}</span>${note}</div>`;
  };

  const allKeys=[...primary,...derived];
  const isModeEntry=!d.mode||d.mode==="entry";

  const r=document.createElement("div");r.className="msg-row ai";
  r.innerHTML=`<div class="msg-avatar" style="background:var(--g0);color:var(--g8);font-size:14px">🎤</div>
    <div class="msg-body">
      <div class="extraction-card">
        <div class="ext-header">🎤 Extracted from voice · ${allKeys.length} field${allKeys.length>1?"s":""}</div>
        <div class="ext-grid">${allKeys.map(renderExtField).join("")}</div>
        ${isModeEntry?`<div style="margin-top:12px;display:flex;gap:8px;flex-wrap:wrap">
          <button class="ra primary" style="font-size:11.5px" onclick="sendFollowUp('Is this child weight normal for the age?')">🩺 Ask Advisory</button>
          <button class="ra" style="font-size:11.5px" onclick="exportFields()">⬇ Export</button>
        </div>`:""}
      </div>
    </div>`;
  document.getElementById("chatFeed").appendChild(r);
}

function addDataMsg(d){
  // Data entry mode: extraction summary is already added by addExtractionSummary()
  // Nothing more to render for entry mode
}

// ── TTS: Sarvam B64 audio OR browser SpeechSynthesis fallback ──────────
function playTTS(id){
  const row=id?document.getElementById(id):null;
  const b64=row?row.dataset.tts:lastTTS;
  if(b64){
    new Audio("data:audio/wav;base64,"+b64).play().catch(e=>speakText(id));
  } else {
    speakText(id);
  }
}
function playLastTTS(){playTTS(null);}

function speakText(id){
  if(!window.speechSynthesis){toast("TTS not available in this browser",2500);return;}
  const row=id?document.getElementById(id):null;
  const text=row?(row.dataset.advText||row.dataset.orig):lastAdvisoryText;
  if(!text){toast("No text to speak",2000);return;}
  window.speechSynthesis.cancel();
  const utt=new SpeechSynthesisUtterance(text.slice(0,500));
  utt.lang= row?(row.dataset.olang==="hi"?"hi-IN":"en-US"):"en-US";
  utt.rate=0.95;utt.pitch=1;
  window.speechSynthesis.speak(utt);
  toast("Speaking…",2000);
}
let lastAdvisoryText="";

// ── Translate toggle ──────────────────────────────────────────────────
const LMAP_FULL={hi:"Hindi",ta:"Tamil",te:"Telugu",mr:"Marathi",bn:"Bengali",
  gu:"Gujarati",kn:"Kannada",ml:"Malayalam",pa:"Punjabi",or:"Odia",as:"Assamese"};
async function xlat(id,ol){
  const r=document.getElementById(id);if(!r)return;
  const ab=document.getElementById("ab_"+id),btn=document.getElementById("xb_"+id);
  if(!ab||!btn)return;
  const langName=LMAP_FULL[ol]||ol;
  if(r.dataset.clang==="en"){
    // currently showing English → switch back to target language
    ab.textContent=r.dataset.orig;r.dataset.clang=ol;
    btn.textContent="↔ English";return;
  }
  // currently showing target language → switch to English
  if(r.dataset.eng&&r.dataset.eng!==r.dataset.orig){
    ab.textContent=r.dataset.eng;r.dataset.clang="en";
    btn.textContent="↩ "+langName;return;
  }
  // fallback: fetch English translation via API
  btn.textContent="…";btn.disabled=true;
  try{
    const rs=await fetch("/api/translate",{method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({text:r.dataset.orig,source:ol,target:"en"})});
    const data=await rs.json();
    r.dataset.eng=data.translated;ab.textContent=data.translated;
    r.dataset.clang="en";btn.textContent="↩ "+langName;
  }catch(e){toast("Translation unavailable — check connection",3000);}
  finally{btn.disabled=false;}
}

// ── Add to Log ──────────────────────────────────────────────────────
const sessionLog=[];
function addToLog(id){
  const r=document.getElementById(id);
  const text=r?r.dataset.orig:"";
  const langCode=r?r.dataset.olang:"";
  sessionLog.push({time:new Date().toLocaleTimeString(),lang:LMAP[langCode]||langCode,text:text.slice(0,300)});
  toast(`Saved to session log (${sessionLog.length} entries)`,2500);
}

function fb(btn,v){
  btn.style.background=v>0?"#d1fae5":"#fee2e2";
  btn.style.borderColor=v>0?"#34d399":"#f87171";
  toast(v>0?"Thank you for the positive feedback! 👍":"Feedback noted — we will improve 🙏",2500);
}

function exportFields(){
  if(!lastFields){toast("No field data to export",2000);return;}
  const clean=Object.fromEntries(Object.entries(lastFields).map(([k,v])=>[k,v.value]));
  const a=document.createElement("a");
  a.href=URL.createObjectURL(new Blob([JSON.stringify(clean,null,2)],{type:"application/json"}));
  a.download="poshan_entry.json";a.click();
  toast("JSON file downloaded",2000);
}

function clearChat(){
  document.getElementById("chatFeed").innerHTML=`<div class="welcome" id="welcomeState">
    <div class="welcome-icon">🌾</div>
    <h2>Poshan AI Advisory System</h2>
    <p>Ask a clinical question in your language and receive structured advisory guidance based on official MoWCD, NHM and ICDS guidelines.</p>
    <div class="welcome-features">
      <div class="welcome-feat"><div class="wf-icon">🎙️</div><div class="wf-lbl">Voice Input</div>12 languages</div>
      <div class="welcome-feat"><div class="wf-icon">📋</div><div class="wf-lbl">WHO Norms</div>Instant advice</div>
      <div class="welcome-feat"><div class="wf-icon">📚</div><div class="wf-lbl">43 PDF guides</div>Official sources</div>
    </div>
    <div class="welcome-chips" id="welcomeChips"></div></div>`;
  buildWelcomeChips();lastTTS=null;lastFields=null;lastAdvisoryText="";
  sessionStats.queries=0;sessionStats.entries=0;
  Object.keys(sessionStats.langs).forEach(k=>delete sessionStats.langs[k]);
  Object.keys(sessionStats.topics).forEach(k=>delete sessionStats.topics[k]);
  window.speechSynthesis&&window.speechSynthesis.cancel();
  // ensure chat panel is visible
  document.getElementById("chatFeed").style.display="";
  document.getElementById("inputPanel").style.display="";
}

async function loadHistory(){
  try{
    const r=await fetch("/api/history");const {history}=await r.json();
    if(!history.length){toast("No history yet",2000);return;}
    const lines=history.slice(0,5).map(h=>`${LMAP[h.lang]||h.lang}: ${(h.transcript||"").slice(0,60)}`).join("\n");
    toast(`Last ${history.length} sessions:\n${lines}`,5000);
  }catch(e){toast("Could not load history",2000);}
}

function esc(s){return String(s||"").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");}
function toast(msg,ms=2000){const t=document.getElementById("toast");t.textContent=msg;t.classList.add("show");setTimeout(()=>t.classList.remove("show"),ms);}

function toggleSteps(id){
  const sr=document.getElementById("sr_"+id),st=document.getElementById("st_"+id);
  if(!sr||!st)return;
  const hidden=sr.style.display==="none";
  sr.style.display=hidden?"":"none";
  const cnt=sr.querySelectorAll("li").length;
  st.textContent=hidden?"▲ Show less":"▼ "+cnt+" more steps";
}

function sendFollowUp(txt){
  const inp=document.getElementById("textInput");
  if(!inp)return;
  inp.value=txt;
  sendText();
}

function switchBnTab(el,tab){
  document.querySelectorAll(".bn-item").forEach(x=>x.classList.remove("active"));
  el.classList.add("active");
  const chatFeed=document.getElementById("chatFeed");
  const inputPanel=document.getElementById("inputPanel");
  const panels=["panelLog","panelAnalytics","panelSettings"];
  if(tab==="chat"){
    chatFeed.style.display="";inputPanel.style.display="";
    panels.forEach(p=>document.getElementById(p).style.display="none");
    return;
  }
  chatFeed.style.display="none";inputPanel.style.display="none";
  panels.forEach(p=>document.getElementById(p).style.display="none");
  if(tab==="log"){document.getElementById("panelLog").style.display="";renderLogPanel();}
  if(tab==="analytics"){document.getElementById("panelAnalytics").style.display="";renderAnalytics();}
  if(tab==="settings"){document.getElementById("panelSettings").style.display="";renderSettings();}
}

// ── Profile / Settings shortcut ───────────────────────────────────────
function openSettings(){
  document.querySelectorAll(".sb-item").forEach(x=>x.classList.remove("active"));
  document.querySelector(".sb-item:nth-child(4)").classList.add("active");
  document.querySelectorAll(".bn-item").forEach((x,i)=>x.classList.toggle("active",i===3));
  document.getElementById("chatFeed").style.display="none";
  document.getElementById("inputPanel").style.display="none";
  ["panelLog","panelAnalytics"].forEach(p=>document.getElementById(p).style.display="none");
  document.getElementById("panelSettings").style.display="";
  renderSettings();
}

// ── Notifications ─────────────────────────────────────────────────────
const notifs=[];
function pushNotif(title,body){
  notifs.unshift({title,body,time:new Date().toLocaleTimeString(),unread:true});
  if(notifs.length>20)notifs.pop();
  const unread=notifs.filter(n=>n.unread).length;
  const badge=document.getElementById("notifBadge");
  badge.textContent=unread;badge.style.display=unread?"":"none";
}
function toggleNotif(e){
  e.stopPropagation();
  const d=document.getElementById("notifDropdown");
  const visible=d.style.display!=="none";
  d.style.display=visible?"none":"";
  if(!visible){
    notifs.forEach(n=>n.unread=false);
    document.getElementById("notifBadge").style.display="none";
    renderNotifList();
  }
}
function renderNotifList(){
  const list=document.getElementById("notifList");
  if(!notifs.length){list.innerHTML=`<div class="notif-empty">No notifications yet</div>`;return;}
  list.innerHTML=notifs.map(n=>`<div class="notif-item${n.unread?" unread":""}">
    <div class="notif-item-title">${esc(n.title)}</div>
    <div class="notif-item-body">${esc(n.body)} · ${n.time}</div></div>`).join("");
}
function clearNotifs(){
  notifs.length=0;
  document.getElementById("notifBadge").style.display="none";
  renderNotifList();
}
// Close dropdown when clicking outside
document.addEventListener("click",e=>{
  const d=document.getElementById("notifDropdown");
  if(d.style.display!=="none"&&!d.contains(e.target)&&e.target.id!=="bellBtn"){
    d.style.display="none";
  }
});
</script>
</body>
</html>"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
