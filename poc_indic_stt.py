"""
POC: Indic Language Speech-to-Text using Open Source / Free Models
Services:
  1. OpenAI Whisper  — open-source, offline, 11 Indic languages
  2. Sarvam AI       — free API, 22 scheduled Indian languages, code-mix
  3. Bhashini ULCA   — Govt. of India free API, 22 scheduled languages

Install:
  pip install openai-whisper sounddevice soundfile requests
"""

import argparse
import base64
import os
import sys
import tempfile
import time

import requests

# ── CONFIG ────────────────────────────────────────────────────────────────────

SARVAM_API_KEY   = os.getenv("SARVAM_API_KEY",   "")
BHASHINI_API_KEY = os.getenv("BHASHINI_API_KEY",  "")
BHASHINI_USER_ID = os.getenv("BHASHINI_USER_ID",  "")

SAMPLE_RATE  = 16000  # Hz — required by Whisper and Sarvam
RECORD_SECS  = 5

INDIC_LANGUAGES = {
    "hi":  "Hindi",      "bn":  "Bengali",   "ta":  "Tamil",
    "te":  "Telugu",     "mr":  "Marathi",   "gu":  "Gujarati",
    "kn":  "Kannada",    "ml":  "Malayalam", "pa":  "Punjabi",
    "as":  "Assamese",   "ur":  "Urdu",      "or":  "Odia",
    "sa":  "Sanskrit",   "mai": "Maithili",  "kok": "Konkani",
}

SARVAM_CODES = {k: f"{k}-IN" for k in INDIC_LANGUAGES}
SARVAM_CODES.update({"mai": "mai-IN", "kok": "kok-IN"})


# ── HELPERS ───────────────────────────────────────────────────────────────────

def _sep(label: str = ""):
    w = 62
    if label:
        pad = (w - len(label) - 2) // 2
        print(f"\n{'─' * pad} {label} {'─' * pad}")
    else:
        print("─" * w)


def _ok(label: str, value: str, extra: str = ""):
    suffix = f"  ({extra})" if extra else ""
    print(f"  {label:<14}: {value}{suffix}")


# ── AUDIO RECORDER ────────────────────────────────────────────────────────────

def record_audio(duration: int = RECORD_SECS) -> str:
    try:
        import sounddevice as sd
        import soundfile as sf
    except ImportError:
        sys.exit("Missing deps — run: pip install sounddevice soundfile")

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()

    print(f"\n  Get ready — recording starts in:", end=" ", flush=True)
    for i in (3, 2, 1):
        print(i, end="... ", flush=True)
        time.sleep(1)
    print("GO!", flush=True)

    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=1, dtype="float32")
    for remaining in range(duration, 0, -1):
        print(f"\r  Recording... {remaining}s left  ", end="", flush=True)
        time.sleep(1)
    sd.wait()
    print(f"\r  Recording done. Saved to {tmp.name}")

    sf.write(tmp.name, audio, SAMPLE_RATE)
    return tmp.name


# ── 1. OPENAI WHISPER ─────────────────────────────────────────────────────────

def transcribe_whisper(audio_path: str, lang: str = "hi",
                       model_size: str = "small", auto_detect: bool = False) -> dict:
    try:
        import whisper
    except ImportError:
        return {"error": "Run: pip install openai-whisper"}

    lang_arg = None if auto_detect else lang
    print(f"\n  [Whisper] Loading '{model_size}' model...", flush=True)
    model  = whisper.load_model(model_size)

    t0     = time.time()
    result = model.transcribe(audio_path, language=lang_arg, task="transcribe")
    elapsed = round(time.time() - t0, 2)

    detected = result.get("language", lang)
    return {
        "service":      "OpenAI Whisper (offline, open-source)",
        "model":        model_size,
        "language":     INDIC_LANGUAGES.get(detected, detected),
        "detected_lang": detected if auto_detect else None,
        "text":         result["text"].strip(),
        "latency_s":    elapsed,
    }


# ── 2. SARVAM AI ──────────────────────────────────────────────────────────────

def transcribe_sarvam(audio_path: str, lang: str = "hi") -> dict:
    if not SARVAM_API_KEY:
        return {"service": "Sarvam AI", "skipped": "Set SARVAM_API_KEY env var — free at dashboard.sarvam.ai"}

    lang_code = SARVAM_CODES.get(lang, f"{lang}-IN")
    url       = "https://api.sarvam.ai/speech-to-text"
    headers   = {"api-subscription-key": SARVAM_API_KEY}

    try:
        with open(audio_path, "rb") as f:
            files = {"file": (os.path.basename(audio_path), f, "audio/wav")}
            data  = {"language_code": lang_code, "model": "saaras:v3",
                     "with_timestamps": "false"}
            t0   = time.time()
            resp = requests.post(url, headers=headers, files=files,
                                 data=data, timeout=30)
            elapsed = round(time.time() - t0, 2)

        resp.raise_for_status()
        return {
            "service":   "Sarvam AI — Saaras v2 (free API, 22 languages)",
            "language":  lang_code,
            "text":      resp.json().get("transcript", ""),
            "latency_s": elapsed,
        }
    except requests.HTTPError as e:
        return {"service": "Sarvam AI", "error": f"HTTP {resp.status_code}: {resp.text[:150]}"}
    except Exception as e:
        return {"service": "Sarvam AI", "error": str(e)}


# ── 3. BHASHINI ULCA ──────────────────────────────────────────────────────────

def transcribe_bhashini(audio_path: str, lang: str = "hi") -> dict:
    if not BHASHINI_API_KEY or not BHASHINI_USER_ID:
        return {"service": "Bhashini ULCA", "skipped": "Set BHASHINI_API_KEY and BHASHINI_USER_ID — free at bhashini.gov.in/ulca"}

    pipeline_url = "https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/getModelsPipeline"
    auth_headers = {
        "userID":     BHASHINI_USER_ID,
        "ulcaApiKey": BHASHINI_API_KEY,
        "Content-Type": "application/json",
    }
    pipeline_payload = {
        "pipelineTasks": [{"taskType": "asr",
                           "config": {"language": {"sourceLanguage": lang}}}],
        "pipelineRequestConfig": {"pipelineId": "64392f96daac500b55c543cd"},
    }

    try:
        pipe = requests.post(pipeline_url, json=pipeline_payload,
                             headers=auth_headers, timeout=15)
        pipe.raise_for_status()
        pipe_data    = pipe.json()
        service_id   = pipe_data["pipelineResponseConfig"][0]["config"][0]["serviceId"]
        callback_url = pipe_data["pipelineInferenceAPIEndPoint"]["callbackUrl"]
        inf_key      = pipe_data["pipelineInferenceAPIEndPoint"]["inferenceApiKey"]["value"]
    except Exception as e:
        return {"service": "Bhashini ULCA", "error": f"Pipeline setup failed: {e}"}

    with open(audio_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode()

    inf_payload = {
        "pipelineTasks": [{
            "taskType": "asr",
            "config": {
                "serviceId": service_id,
                "language":  {"sourceLanguage": lang},
                "audioFormat": "wav",
                "samplingRate": SAMPLE_RATE,
            },
        }],
        "inputData": {"audio": [{"audioContent": audio_b64}]},
    }

    try:
        t0   = time.time()
        resp = requests.post(callback_url, json=inf_payload,
                             headers={"Authorization": inf_key,
                                      "Content-Type": "application/json"},
                             timeout=30)
        elapsed = round(time.time() - t0, 2)
        resp.raise_for_status()
        text = resp.json()["pipelineResponse"][0]["output"][0]["source"]
        return {
            "service":   "Bhashini ULCA (Govt. of India, free)",
            "language":  lang,
            "text":      text,
            "latency_s": elapsed,
        }
    except Exception as e:
        return {"service": "Bhashini ULCA", "error": str(e)}


# ── DISPLAY ───────────────────────────────────────────────────────────────────

def print_result(r: dict):
    svc = r.get("service", "Unknown")
    _sep(svc)
    if "skipped" in r:
        print(f"  SKIPPED — {r['skipped']}")
    elif "error" in r:
        print(f"  ERROR   — {r['error']}")
    else:
        _ok("Transcript", r.get("text") or "(empty)")
        if r.get("detected_lang"):
            _ok("Detected lang", r["detected_lang"])
        if "latency_s" in r:
            _ok("Latency", f"{r['latency_s']}s")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="POC: Indic Language STT — Whisper, Sarvam AI, Bhashini")
    parser.add_argument("audio",    nargs="?",  help="Path to WAV file (omit to record from mic)")
    parser.add_argument("--lang",   default="hi", choices=list(INDIC_LANGUAGES),
                        help="Language code (default: hi = Hindi)")
    parser.add_argument("--model",  default="small",
                        choices=["tiny", "base", "small", "medium", "large-v3"],
                        help="Whisper model size (default: small)")
    parser.add_argument("--auto",   action="store_true",
                        help="Let Whisper auto-detect the language")
    parser.add_argument("--list",   action="store_true",
                        help="List supported language codes and exit")
    args = parser.parse_args()

    if args.list:
        print("\nSupported language codes:\n")
        for code, name in INDIC_LANGUAGES.items():
            print(f"  {code:<6} {name}")
        print()
        return

    lang_name = INDIC_LANGUAGES.get(args.lang, args.lang)

    print("\n" + "═" * 62)
    print("  POC — Indic Language Speech-to-Text (Open Source / Free)")
    print("═" * 62)
    print(f"  Language : {lang_name} ({args.lang})")
    print(f"  Whisper  : {args.model} model {'[auto-detect]' if args.auto else ''}")
    print(f"  Sarvam   : {'API key set ✓' if SARVAM_API_KEY else 'no key (will skip)'}")
    print(f"  Bhashini : {'API key set ✓' if BHASHINI_API_KEY else 'no key (will skip)'}")
    print("═" * 62)

    audio_path = args.audio or record_audio()

    results = [
        transcribe_whisper(audio_path, args.lang, args.model, args.auto),
        transcribe_sarvam(audio_path, args.lang),
        transcribe_bhashini(audio_path, args.lang),
    ]

    _sep("RESULTS")
    for r in results:
        print_result(r)

    print("\n" + "═" * 62 + "\n")

    if args.audio is None and os.path.exists(audio_path):
        os.unlink(audio_path)


if __name__ == "__main__":
    main()
