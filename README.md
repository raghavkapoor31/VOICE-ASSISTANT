# Indic Language STT — POC

Proof of concept for open-source Indic language speech-to-text, relevant to the Poshan AI voice assistant.

## Services covered

| Service | Type | Indic Languages | Cost |
|---|---|---|---|
| OpenAI Whisper | Open-source, offline | 11 | Free |
| Sarvam AI (Saaras v3) | Free API | 22 (all scheduled) | Free |
| Bhashini ULCA | Govt. of India API | 22 (all scheduled) | Free |

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
# Record from mic (5 seconds), transcribe in Hindi
python poc_indic_stt.py

# Use an existing WAV file, Tamil
python poc_indic_stt.py audio.wav ta

# Use an existing WAV file, Telugu
python poc_indic_stt.py audio.wav te
```

## API Keys (optional — Whisper works without any key)

```bash
export SARVAM_API_KEY=your_key      # https://dashboard.sarvam.ai
export BHASHINI_API_KEY=your_key    # https://bhashini.gov.in/ulca
export BHASHINI_USER_ID=your_id
```

## Supported language codes

| Code | Language |
|---|---|
| hi | Hindi |
| bn | Bengali |
| ta | Tamil |
| te | Telugu |
| mr | Marathi |
| gu | Gujarati |
| kn | Kannada |
| ml | Malayalam |
| pa | Punjabi |
| as | Assamese |
| ur | Urdu |
# VOICE-ASSISTANT
# VOICE-ASSISTANT
