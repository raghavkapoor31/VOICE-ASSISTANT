from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak
)
from reportlab.graphics.shapes import (
    Drawing, Rect, String, Arrow, Line, Polygon, Group
)
from reportlab.graphics import renderPDF

W, H = A4
OUT = "/Users/raghav/Desktop/Poshan ai Doc/tech_arch.pdf"

doc = SimpleDocTemplate(OUT, pagesize=A4,
                        leftMargin=1.8*cm, rightMargin=1.8*cm,
                        topMargin=1.8*cm, bottomMargin=1.8*cm)

def sty(name, **kw):
    s = getSampleStyleSheet()
    return ParagraphStyle(name, parent=s['Normal'], **kw)

TITLE  = sty('T',  fontSize=16, alignment=TA_CENTER, fontName='Helvetica-Bold',
              textColor=colors.HexColor('#0d1b2a'), spaceAfter=2)
SUB    = sty('SB', fontSize=9,  alignment=TA_CENTER, textColor=colors.HexColor('#555'),
              spaceAfter=8)
H2     = sty('H2', fontSize=12, fontName='Helvetica-Bold', spaceBefore=12, spaceAfter=5,
              textColor=colors.HexColor('#1a237e'))
H3     = sty('H3', fontSize=10, fontName='Helvetica-Bold', spaceBefore=6, spaceAfter=3,
              textColor=colors.HexColor('#283593'))
BODY   = sty('BD', fontSize=9,  leading=14, alignment=TA_JUSTIFY)
NOTE   = sty('NT', fontSize=8,  leading=12, textColor=colors.HexColor('#666'),
              leftIndent=6)
MONO   = sty('MN', fontSize=8,  fontName='Courier', leading=12,
              textColor=colors.HexColor('#1b5e20'), leftIndent=10)

# ── ARCH DIAGRAM ─────────────────────────────────────────────────────────────

def arch_diagram():
    """Draw the end-to-end voice assistant pipeline as a flow diagram."""
    dw, dh = 510, 230
    d = Drawing(dw, dh)

    # background
    d.add(Rect(0, 0, dw, dh, rx=6, ry=6,
               fillColor=colors.HexColor('#f8f9fa'), strokeColor=colors.HexColor('#dee2e6'),
               strokeWidth=1))

    boxes = [
        # (x, y, w, h, label_line1, label_line2, bg)
        (10,  90, 75, 50, "ANGANWADI",    "WORKER",      '#1565c0'),
        (105, 90, 75, 50, "VOICE",        "CAPTURE",     '#1976d2'),
        (200, 90, 75, 50, "STT ENGINE",   "Whisper/Sarvam/\nBhashini", '#2e7d32'),
        (295, 90, 75, 50, "NLP + LLM",   "Intent &\nSlot Extract",  '#6a1b9a'),
        (390, 90, 75, 50, "POSHAN",       "TRACKER API", '#e65100'),
        (200, 10, 75, 50, "TTS ENGINE",   "Voice\nResponse",         '#00695c'),
    ]

    def draw_box(x, y, w, h, l1, l2, bg):
        d.add(Rect(x, y, w, h, rx=5, ry=5,
                   fillColor=colors.HexColor(bg),
                   strokeColor=colors.white, strokeWidth=1.5))
        d.add(String(x + w/2, y + h - 14, l1,
                     fontName='Helvetica-Bold', fontSize=7.5,
                     fillColor=colors.white, textAnchor='middle'))
        # second label line(s)
        for i, line in enumerate(l2.split('\n')):
            d.add(String(x + w/2, y + h - 26 - i*10, line,
                         fontName='Helvetica', fontSize=6.5,
                         fillColor=colors.HexColor('#e0e0e0'), textAnchor='middle'))

    for bx, by, bw, bh, l1, l2, bg in boxes:
        draw_box(bx, by, bw, bh, l1, l2, bg)

    # arrows between horizontal boxes (y=90 row)
    arrow_y = 115
    xs = [85, 180, 275, 370]
    for ax in xs:
        d.add(Line(ax, arrow_y, ax + 18, arrow_y,
                   strokeColor=colors.HexColor('#555'), strokeWidth=1.5))
        d.add(Polygon([ax+18, arrow_y+4, ax+18, arrow_y-4, ax+26, arrow_y],
                      fillColor=colors.HexColor('#555'), strokeColor=colors.HexColor('#555')))

    # TTS feedback arrow up from NLP box (295+37=332) down to TTS box
    d.add(Line(237, 90, 237, 60,
               strokeColor=colors.HexColor('#00695c'), strokeWidth=1.5,
               strokeDashArray=[3, 2]))
    d.add(Polygon([233, 62, 241, 62, 237, 58],
                  fillColor=colors.HexColor('#00695c'),
                  strokeColor=colors.HexColor('#00695c')))

    # arrow from NLP to TTS (dashed feedback)
    d.add(Line(332, 115, 332, 35, strokeColor=colors.HexColor('#00695c'),
               strokeWidth=1.5, strokeDashArray=[3, 2]))
    d.add(Line(332, 35, 275, 35, strokeColor=colors.HexColor('#00695c'),
               strokeWidth=1.5, strokeDashArray=[3, 2]))
    d.add(Polygon([277, 31, 277, 39, 273, 35],
                  fillColor=colors.HexColor('#00695c'),
                  strokeColor=colors.HexColor('#00695c')))

    # label arrows
    arrow_labels = [
        (88,  122, "16kHz WAV"),
        (183, 122, "Text"),
        (278, 122, "Intent+Slots"),
        (373, 122, "API Call"),
        (248, 75,  "TTS text"),
    ]
    for lx, ly, lt in arrow_labels:
        d.add(String(lx, ly, lt, fontName='Helvetica', fontSize=6,
                     fillColor=colors.HexColor('#888'), textAnchor='middle'))

    # bottom legend
    legend = [
        ('#1565c0', 'User'),
        ('#2e7d32', 'Open-source STT'),
        ('#6a1b9a', 'AI/NLP Layer'),
        ('#e65100', 'Govt. Backend'),
        ('#00695c', 'TTS Feedback'),
    ]
    lx = 15
    for lc, lt in legend:
        d.add(Rect(lx, 4, 8, 8, fillColor=colors.HexColor(lc),
                   strokeColor=colors.white))
        d.add(String(lx + 11, 5, lt, fontName='Helvetica', fontSize=6.5,
                     fillColor=colors.HexColor('#444')))
        lx += 90

    return d


# ── BUILD PDF ─────────────────────────────────────────────────────────────────

story = []

story.append(Paragraph("Technical Architecture Brief", TITLE))
story.append(Paragraph("Poshan AI Voice Assistant — End-to-End System Design", SUB))
story.append(HRFlowable(width="100%", thickness=1,
                         color=colors.HexColor('#1a237e'), spaceAfter=8))

story.append(Paragraph("System Overview", H2))
story.append(Paragraph(
    "The Poshan AI Voice Assistant enables Anganwadi workers to fill nutrition tracking forms "
    "hands-free by speaking in their native Indian language. Audio is captured on a low-end "
    "Android device, converted to text via an open-source/free STT engine, parsed by an LLM "
    "for intent and slot extraction, and submitted directly to the Poshan Tracker API — "
    "with a spoken confirmation played back via TTS.", BODY))
story.append(Spacer(1, 0.3*cm))

# Arch diagram
story.append(Paragraph("Architecture Diagram", H3))
story.append(arch_diagram())
story.append(Spacer(1, 0.2*cm))

# ── COMPONENT TABLE ───────────────────────────────────────────────────────────
story.append(Paragraph("Component Breakdown", H2))

comp_data = [
    ["Layer", "Component", "Technology / Tool", "Why"],
    ["Voice Capture",
     "Mic Input\n& Pre-processing",
     "Android MediaRecorder\n16kHz mono WAV",
     "Minimum quality for Indic STT accuracy"],
    ["STT Engine\n(Open Source)",
     "Speech-to-Text",
     "OpenAI Whisper (offline)\nSarvam Saaras v2 (API)\nBhashini ULCA (Govt API)",
     "11–22 Indic languages;\nBhashini is free for Govt apps"],
    ["NLP / AI Layer",
     "Intent & Slot\nExtraction",
     "LLM (Claude / GPT-4o mini)\nor fine-tuned IndicBERT",
     "Extracts child name, weight,\nage, vaccination from text"],
    ["Integration",
     "Poshan Tracker\nAPI",
     "REST API (MeitY)\nOAuth2 / HMAC auth",
     "Directly submits to national\nnutrition database"],
    ["TTS Feedback",
     "Voice Confirmation",
     "Sarvam Bulbul TTS\nor gTTS (offline fallback)",
     "Confirms entry in worker's\nown language"],
    ["Device",
     "Target Hardware",
     "Android 8+ smartphone\n(≥2GB RAM)",
     "Matches avg Anganwadi\nworker device profile"],
]

ct = Table(comp_data, colWidths=[2.5*cm, 3*cm, 4.5*cm, 5*cm])
ct.setStyle(TableStyle([
    ('BACKGROUND',   (0,0), (-1,0), colors.HexColor('#1a237e')),
    ('TEXTCOLOR',    (0,0), (-1,0), colors.white),
    ('FONTNAME',     (0,0), (-1,0), 'Helvetica-Bold'),
    ('FONTSIZE',     (0,0), (-1,-1), 8.5),
    ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#e8eaf6'), colors.white]),
    ('GRID',         (0,0), (-1,-1), 0.4, colors.HexColor('#9fa8da')),
    ('VALIGN',       (0,0), (-1,-1), 'TOP'),
    ('ALIGN',        (0,0), (-1,-1), 'LEFT'),
    ('TOPPADDING',   (0,0), (-1,-1), 5),
    ('BOTTOMPADDING',(0,0), (-1,-1), 5),
    ('LEFTPADDING',  (0,0), (-1,-1), 5),
]))
story.append(ct)
story.append(Spacer(1, 0.4*cm))

# ── DATA FLOW ─────────────────────────────────────────────────────────────────
story.append(Paragraph("Data Flow — Step by Step", H2))

flow_data = [
    ["Step", "Action", "Input → Output"],
    ["1", "Worker speaks into phone",
     "Natural speech (Hindi / Telugu / etc.) → 16kHz WAV"],
    ["2", "STT transcription",
     "WAV audio → raw Indic text transcript"],
    ["3", "NLP slot extraction",
     "Raw text → structured JSON\n{child_name, weight_kg, age_months, village}"],
    ["4", "Validation prompt",
     "Structured data → TTS reads back: \"Aarav, 8.2 kg, 14 months — confirm?\""],
    ["5", "Worker confirms (\"Haan\" / \"Yes\")",
     "Voice confirmation → trigger API call"],
    ["6", "API submission",
     "JSON payload → Poshan Tracker REST API → HTTP 200 OK"],
    ["7", "TTS acknowledgement",
     "Success flag → \"Data saved successfully\" in local language"],
]

ft = Table(flow_data, colWidths=[1.2*cm, 4*cm, 9.8*cm])
ft.setStyle(TableStyle([
    ('BACKGROUND',    (0,0), (-1,0), colors.HexColor('#263238')),
    ('TEXTCOLOR',     (0,0), (-1,0), colors.white),
    ('FONTNAME',      (0,0), (-1,0), 'Helvetica-Bold'),
    ('FONTSIZE',      (0,0), (-1,-1), 8.5),
    ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#eceff1'), colors.white]),
    ('GRID',          (0,0), (-1,-1), 0.4, colors.grey),
    ('VALIGN',        (0,0), (-1,-1), 'TOP'),
    ('ALIGN',         (0,0), (0,-1), 'CENTER'),
    ('ALIGN',         (1,0), (-1,-1), 'LEFT'),
    ('TOPPADDING',    (0,0), (-1,-1), 4),
    ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ('LEFTPADDING',   (0,0), (-1,-1), 5),
    ('FONTNAME',      (2,1), (2,-1), 'Courier'),
    ('FONTSIZE',      (2,1), (2,-1), 8),
]))
story.append(ft)
story.append(Spacer(1, 0.4*cm))

# ── TECH STACK ────────────────────────────────────────────────────────────────
story.append(Paragraph("Technology Stack Summary", H2))

stack_data = [
    ["Category", "Open Source / Free Option", "Indic Languages", "Deployment"],
    ["STT (offline)", "OpenAI Whisper large-v3", "11", "On-device / server"],
    ["STT (API)",     "Sarvam AI Saaras v2",     "22", "Cloud API (free tier)"],
    ["STT (Govt)",    "Bhashini ULCA ASR",        "22", "Cloud API (free)"],
    ["TTS",           "Sarvam Bulbul / gTTS",     "11 / 1", "Cloud API / offline"],
    ["NLP / LLM",     "Claude Haiku / IndicBERT", "All", "Cloud / fine-tuned"],
    ["Backend",       "FastAPI + PostgreSQL",      "—", "Self-hosted / GCP"],
    ["Mobile App",    "Flutter (Android)",         "—", "Android 8+"],
]

st = Table(stack_data, colWidths=[3.5*cm, 5*cm, 3*cm, 3.5*cm])
st.setStyle(TableStyle([
    ('BACKGROUND',    (0,0), (-1,0), colors.HexColor('#37474f')),
    ('TEXTCOLOR',     (0,0), (-1,0), colors.white),
    ('FONTNAME',      (0,0), (-1,0), 'Helvetica-Bold'),
    ('FONTSIZE',      (0,0), (-1,-1), 8.5),
    ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#fafafa'), colors.white]),
    ('GRID',          (0,0), (-1,-1), 0.4, colors.grey),
    ('ALIGN',         (0,0), (-1,-1), 'LEFT'),
    ('ALIGN',         (2,0), (2,-1), 'CENTER'),
    ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
    ('TOPPADDING',    (0,0), (-1,-1), 4),
    ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ('LEFTPADDING',   (0,0), (-1,-1), 5),
]))
story.append(st)
story.append(Spacer(1, 0.3*cm))

story.append(HRFlowable(width="100%", thickness=0.5,
                         color=colors.HexColor('#ccc'), spaceAfter=4))
story.append(Paragraph(
    "POC code available: poc_indic_stt.py — demonstrates Whisper, Sarvam AI, and Bhashini "
    "running in parallel on the same audio input with latency benchmarks.", NOTE))

doc.build(story)
print("Done:", OUT)
