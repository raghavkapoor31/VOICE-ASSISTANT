#!/usr/bin/env python3
"""
PDF Ingestion Pipeline — Poshan AI
Extracts text from all official MoWCD / NHM / ICDS PDFs, chunks them,
embeds with paraphrase-multilingual-MiniLM-L12-v2, and saves a merged
FAISS index + JSON metadata alongside the manual knowledge base.

Run once (takes ~5–15 min for 43 PDFs):
    cd poc_voice_app && python3 pdf_ingest.py

Outputs:
    pdf_faiss.index  — FAISS flat-IP index of all PDF chunks
    pdf_chunks.json  — chunk metadata (id, title, source, page, content)
"""

import fitz        # PyMuPDF
import json, os, re, sys, time
from pathlib import Path

DOCS_DIR   = Path(__file__).parent.parent / "docs for poshan"
OUT_DIR    = Path(__file__).parent
OUT_INDEX  = OUT_DIR / "pdf_faiss.index"
OUT_CHUNKS = OUT_DIR / "pdf_chunks.json"

CHUNK_SIZE = 480   # target chars per chunk
OVERLAP    = 80    # overlap between consecutive chunks
MIN_CHARS  = 90    # discard chunks shorter than this
MAX_PAGES  = 120   # skip PDFs with more pages than this (likely reports/appendices)
SKIP_PAGES_FRONT = 2   # skip cover pages / blank pages at start
SKIP_PAGES_BACK  = 2   # skip blank back-matter pages

# ── TEXT CLEANING ─────────────────────────────────────────────────────────────

_NOISE = re.compile(
    r'(Page \d+\s*of\s*\d+'
    r'|\d+\s*\|\s*P\s*a\s*g\s*e'
    r'|www\.\S+'
    r'|http\S+'
    r'|©.*?reserved'
    r'|Ministry of.*?Development'   # repetitive headers
    r'|National Health Mission'     # repetitive headers
    r')', re.IGNORECASE
)

def clean(text: str) -> str:
    text = _NOISE.sub(' ', text)
    text = re.sub(r'\s{3,}', '  ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ── CHUNKING ──────────────────────────────────────────────────────────────────

def chunk_text(text: str) -> list[str]:
    sentences = re.split(r'(?<=[.!?।])\s+', text)
    chunks, buf = [], ""
    for sent in sentences:
        if len(buf) + len(sent) + 1 <= CHUNK_SIZE:
            buf = (buf + " " + sent).strip()
        else:
            if len(buf) >= MIN_CHARS:
                chunks.append(buf)
            # carry overlap into next chunk
            words = buf.split()
            overlap_words = words[max(0, len(words)-8):]
            buf = " ".join(overlap_words) + " " + sent.strip()
    if len(buf) >= MIN_CHARS:
        chunks.append(buf)
    return chunks


# ── PDF TITLE NORMALISATION ───────────────────────────────────────────────────

_ABBR = {
    "NHM": "NHM", "MoWCD": "MoWCD", "ICMR": "ICMR", "NIN": "NIN",
    "FSSAI": "FSSAI", "IMNCI": "IMNCI", "RKSK": "RKSK", "WIFS": "WIFS",
    "PMMVY": "PMMVY", "SAM": "SAM", "MAM": "MAM", "IFA": "IFA",
    "RDA": "RDA", "SNP": "SNP", "VHSND": "VHSND", "RMNCHA": "RMNCHA",
}

def pdf_title(path: Path) -> str:
    parts = re.split(r'[_\-]', path.stem)
    words = []
    for p in parts:
        up = p.upper()
        words.append(_ABBR.get(up, p.capitalize()))
    return " ".join(words)


# ── SINGLE PDF EXTRACTION ─────────────────────────────────────────────────────

def extract_pdf(path: Path) -> list[dict]:
    try:
        doc = fitz.open(str(path))
    except Exception as e:
        print(f"    SKIP (open failed): {e}")
        return []

    n_pages = len(doc)
    if n_pages > MAX_PAGES:
        print(f"    SKIP ({n_pages} pages > {MAX_PAGES} limit)")
        doc.close()
        return []

    title = pdf_title(path)
    results = []

    start_p = min(SKIP_PAGES_FRONT, n_pages)
    end_p   = max(0, n_pages - SKIP_PAGES_BACK)

    for page_num in range(start_p, end_p):
        page = doc[page_num]
        raw  = page.get_text("text")
        text = clean(raw)
        if len(text) < MIN_CHARS:
            continue
        for i, chunk in enumerate(chunk_text(text)):
            results.append({
                "id":      f"{path.stem}_p{page_num+1}_c{i}",
                "title":   title,
                "source":  path.name,
                "page":    page_num + 1,
                "content": chunk,
            })

    doc.close()
    return results


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    pdfs = sorted(DOCS_DIR.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs found in {DOCS_DIR}")

    print(f"\n{'='*60}")
    print(f"  Poshan AI — PDF Ingestion Pipeline")
    print(f"  Source  : {DOCS_DIR}")
    print(f"  PDFs    : {len(pdfs)}")
    print(f"{'='*60}\n")

    all_chunks: list[dict] = []
    t_start = time.time()

    for pdf in pdfs:
        t0 = time.time()
        chunks = extract_pdf(pdf)
        all_chunks.extend(chunks)
        status = f"{len(chunks):4d} chunks" if chunks else "   SKIPPED"
        print(f"  [{status}]  {pdf.name}  ({time.time()-t0:.1f}s)")

    print(f"\n  Total chunks extracted: {len(all_chunks)}")
    print(f"  Extraction time      : {time.time()-t_start:.1f}s")

    # ── Save raw chunks ────────────────────────────────────────────────────
    OUT_CHUNKS.write_text(
        json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n  Saved chunk metadata → {OUT_CHUNKS}")

    # ── Embed + FAISS ──────────────────────────────────────────────────────
    print("\n  Loading embedding model...")
    try:
        from sentence_transformers import SentenceTransformer
        import faiss
        import numpy as np
    except ImportError as e:
        sys.exit(f"Missing dependency: {e}\n  pip3 install sentence-transformers faiss-cpu")

    emb = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    texts = [f"{c['title']}\n{c['content']}" for c in all_chunks]
    print(f"  Embedding {len(texts)} chunks (batch size 64)...")
    t0   = time.time()
    vecs = emb.encode(
        texts, batch_size=64, show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")

    faiss.normalize_L2(vecs)
    idx = faiss.IndexFlatIP(vecs.shape[1])
    idx.add(vecs)
    faiss.write_index(idx, str(OUT_INDEX))

    embed_time = time.time() - t0
    print(f"\n  Saved FAISS index → {OUT_INDEX}")
    print(f"  Embedding time    : {embed_time:.1f}s")
    print(f"  Index size        : {idx.ntotal} vectors  dim={vecs.shape[1]}")
    print(f"\n{'='*60}")
    print(f"  Done. Total time: {time.time()-t_start:.1f}s")
    print(f"  Restart poc_voice_app to use the new index.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
