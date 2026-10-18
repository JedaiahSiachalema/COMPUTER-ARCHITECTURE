# Group 90: Computer Architecture Book-Grounded Chatbot

Part of the 130-Group Collaborative Transformer Chatbot Project.

- **Group ID:** G090
- **Subject:** Computer Architecture
- **Institution:** Mulungushi University, Zambia
- **Members:** Mupelwa Sichilima (Billy), Siachalema Jedaiah
- **Books:** 18 PDFs listed in `data/book_manifest.json` (minimum required: 15)

## What this project does

A retrieval-augmented (RAG) chatbot that answers Computer Architecture questions from our 18 textbooks. It retrieves relevant passages, generates an answer from them, shows citations (book and page) that map to real stored passages, and says so when the books do not contain enough evidence.

## Team roles

This is a proposed split. Edit names or tasks if we agree on something different.

### Billy (Mupelwa Sichilima): data pipeline and retrieval

- PDF ingestion with PyMuPDF, manifest validation and the ingestion report (`src/ingest.py`, `evaluation/ingestion_report.json`). **Done.**
- Text cleaning: watermarks, headers and footers, hyphenation (`src/clean.py`). **Written, needs review on real output.**
- Chunking with the standard passage schema and chunk IDs (`src/chunk.py`). **To do.**
- Embeddings and FAISS index using the approved model and pinned revision (`src/index.py`). **To do.**
- `retrieve(question, top_k=5)` (`src/retrieve.py`). **To do.**
- Tests for ingestion, chunking and retrieval (`tests/`). **To do.**

### Jedaiah (Siachalema Jedaiah): generation, interface and evaluation

- Answer generation from retrieved passages, with a prompt that forbids invented citations (`src/generate.py`). **To do.**
- Citation validation, so every source resolves to a stored passage, and insufficient-evidence behaviour (`src/generate.py`, `tests/`). **To do.**
- Streamlit app (`src/app.py`). **To do.**
- Evaluation questions (30 to 50) with reference answers and expected pages (`data/evaluation_questions.json`). **To do.**
- Evaluation report covering Recall@5, answer correctness, citation correctness, abstention, speed and RAM (`evaluation/report.json`). **To do.**
- Tests for answer format and no-answer questions (`tests/`). **To do.**

### Shared

- **Book manifest:** Billy completes permission status, bibliographic details and source type for books 001 to 009. Jedaiah does books 010 to 018.
- **Review:** every module is checked by the other member before it is marked done.
- **Evaluation questions:** Jedaiah writes them, and Billy checks each reference answer against the cited PDF page, so no answer is reviewed only by its author.
- **README, demo and final submission package:** written together. Billy assembles, and Jedaiah checks it against the submission checklist.

## The interfaces between our parts

Billy's retrieval output feeds Jedaiah's generation, so both of us build to the project's standard interfaces.

**Passage record.** Every chunk keeps these fields: `group_id`, `subject`, `book_id`, `title`, `edition`, `page`, `chunk_id`, `text`. Chunk IDs follow the pattern `G090_BOOK_003_P0143_C02`.

**Retrieval.** `retrieve(question: str, top_k: int = 5) -> list` returns passages with `text`, `score`, `book_id`, `title`, `page` and `chunk_id`.

**Answer.**

```json
{
  "answer": "...",
  "sources": [{"book_id": "G090_BOOK_003", "page": 143, "chunk_id": "G090_BOOK_003_P0143_C02"}],
  "insufficient_evidence": false
}
```

Page numbers are PDF positions. Printed book page numbers have not been verified.

## Setup

Requires Python 3.11 or newer, because `ingest.py` uses `hashlib.file_digest`.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
python -m pip install pymupdf pyyaml
```

TODO: replace this with a pinned `requirements.txt`, and record the approved embedding model and revision in `config/settings.yaml`.

Put the book PDFs in a `books/` folder in the project root. They are not included in this repository.

## Running the pipeline

```bash
python src/ingest.py      # PDFs to data/extracted/*.pages.jsonl
python src/clean.py       # data/cleaned/*.pages.jsonl
# next: chunk, index, retrieve, generate, app
```

## Repository layout

```
README.md
requirements.txt
config/settings.yaml
books/                      # PDFs, not committed
data/
  book_manifest.json
  evaluation_questions.json
  extracted/  cleaned/      # generated, not committed
src/
  ingest.py  clean.py  chunk.py  retrieve.py  generate.py  app.py
tests/
evaluation/
  ingestion_report.json  cleaning_report.json  report.json
```

## Copyright and sharing

The book PDFs, extracted text and indexes are not committed or shared publicly. `books/`, `data/extracted/` and `data/cleaned/` are listed in `.gitignore`. The permission status of each book is recorded in the manifest and is currently `unverified`. TODO: update each entry before submission.

## Known limitations

- Extraction reads existing text layers only. Figures, diagrams and unencoded equations are not captured, and no OCR is used.
- Some pages have fragmented word spacing (mainly the Patt/Patel book) and are flagged for review.
- Cleaning heuristics can remove or merge text incorrectly, so check `evaluation/cleaning_report.json`.
- Evaluation results are not yet available. This section will be updated with actual measurements.
