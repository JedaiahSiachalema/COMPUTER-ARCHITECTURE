# Group 90: Computer Architecture Book-Grounded Chatbot

Part of the 130-Group Collaborative Transformer Chatbot Project.

- **Group ID:** G090
- **Subject:** Computer Architecture
- **Institution:** Mulungushi University, Zambia
- **Members:** Mupelwa Sichilima (Billy), Siachalema Jedaiah
- **Books:** 18 PDFs listed in `data/book_manifest.json` (minimum required: 15)

## What this project does

A retrieval-augmented generation (RAG) chatbot being built to answer Computer Architecture questions from our 18 textbooks. The data pipeline and passage retrieval are implemented. Answer generation, the interface and formal evaluation are still to be implemented.

The current system uses a pretrained transformer to turn textbook passages and questions into embeddings, then searches a FAISS index for relevant passages. Building this index does not train or fine-tune the transformer. No fine-tuning has been completed.

## Current progress

As of 18 October 2026:

- Ingested all 18 books: 14,237 PDF pages, with no failed books.
- Generated 26,238 chunks from usable cleaned pages.
- Corpus preparation removed 604 front-matter chunks and merged 1,044 exact duplicate copies while retaining their source references.
- Indexed 24,590 unique passages with 25,634 original chunk references across all 18 books.
- Retrieval checks passed for three example questions. Billy also manually verified the leading data-hazard result against its PDF page.

These checks establish that retrieval works for the examples checked. Formal retrieval and answer-quality evaluation remains outstanding.

## Team roles

Responsibilities are split as follows. Implementation status does not replace the other member's review.

### Billy (Mupelwa Sichilima): data pipeline and retrieval

- PDF ingestion with PyMuPDF, manifest validation and the ingestion report (`src/ingest.py`, `evaluation/ingestion_report.json`). **Implemented and run on all books.**
- Text cleaning: watermarks, headers, footers and hyphenation, with unusable pages excluded (`src/clean.py`). **Implemented; real output reviewed.**
- Token-based, single-page chunking with the standard passage schema and chunk IDs (`src/chunk.py`). **Implemented and run.**
- Corpus filtering and exact deduplication with all original source references retained (`src/prepare_corpus.py`). **Implemented and run.**
- Embeddings and FAISS index using the configured model and pinned revision (`src/embed.py`). **Implemented and run.**
- `retrieve(question, top_k=5)` (`src/retrieve.py`). **Implemented; example searches and a manual source check passed.**
- CUDA-first model execution with CPU fallback (`src/runtime_device.py`). **Implemented and checked.**
- Tests for ingestion, chunking, corpus preparation, device selection and retrieval (`tests/`). **Written; final full test run and resolution of earlier ingestion test failures remain outstanding.**

Billy's main implementation is complete. Final testing and Jedaiah's module review remain before sign-off.

### Jedaiah (Siachalema Jedaiah): generation, interface and evaluation

- Answer generation from retrieved passages, with a prompt that forbids invented citations (`src/generate.py`). **To do.**
- Citation validation, so every source resolves to a stored passage, and insufficient-evidence behaviour (`src/generate.py`, `tests/`). **To do.**
- Streamlit app (`src/app.py`). **To do.**
- Evaluation questions (30 to 50) with reference answers and expected pages (`data/evaluation_questions.json`). **To do.**
- Evaluation report covering Recall@5, answer correctness, citation correctness, abstention, speed and RAM (`evaluation/report.json`). **To do.**
- Tests for answer format and no-answer questions (`tests/`). **To do.**

### Shared

- **Book manifest:** Billy maintains bibliographic details and source type for books 001 to 009. Jedaiah does books 010 to 018. The pipeline processes all 18 books.
- **Review:** every module is checked by the other member before it is marked done.
- **Evaluation questions:** Jedaiah writes them, and Billy checks each reference answer against the cited PDF page, so no answer is reviewed only by its author.
- **README, demo and final submission package:** written together. Billy assembles, and Jedaiah checks it against the submission checklist.

## The interfaces between our parts

Billy's retrieval output feeds Jedaiah's generation, so both of us build to the project's standard interfaces.

**Passage record.** Every chunk keeps these fields: `group_id`, `subject`, `book_id`, `title`, `edition`, `page`, `chunk_id`, `text`. Chunk IDs follow the pattern `G090_BOOK_003_P0143_C02`.

**Retrieval.** `retrieve(question: str, top_k: int = 5) -> list` returns passages with `text`, `score`, `book_id`, `title`, `page` and `chunk_id`. Results also include `rank`, the full stored metadata and a `sources` list retaining references to duplicate copies.

```python
from src.retrieve import retrieve

passages = retrieve("What is a data hazard in an instruction pipeline?", top_k=5)
```

The model and index are cached within the running Python process. Retrieval uses the model revision and query prefix recorded with the saved index. It returns source passages without generating an answer or deciding whether there is enough evidence. Similarity scores are cosine similarities, not probabilities of correctness.

**Planned answer interface (Jedaiah's part).**

```json
{
  "answer": "...",
  "sources": [{"book_id": "G090_BOOK_003", "page": 143, "chunk_id": "G090_BOOK_003_P0143_C02"}],
  "insufficient_evidence": false
}
```

Page numbers are PDF positions. Printed book page numbers have not been verified.

## Setup

Billy's existing environment is Python 3.12.10 at `C:\Users\Billy\Documents\.venv\Scripts\python.exe`. Reuse it; creating another environment is unnecessary. Commands below are PowerShell commands run from the project root.

```powershell
$projectPython = 'C:\Users\Billy\Documents\.venv\Scripts\python.exe'
```

Dependencies are pinned in `requirements.txt` and are already installed in Billy's environment. If setting up another device, use its existing Python environment and install the requirements there:

```powershell
& $projectPython -m pip install -r requirements.txt
```

The requirements target the current Windows/Python 3.12 setup, including PyTorch 2.7.1 with CUDA 11.8 for the Quadro M2000M. Other platforms or GPUs may need a different compatible PyTorch build. `faiss-cpu` performs index search on the CPU.

### Embedding configuration

| Setting | Current value |
| --- | --- |
| Model | `BAAI/bge-small-en-v1.5` |
| Pinned revision | `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` |
| Embedding dimensions | 384 |
| Model input limit | 512 tokens |
| Chunk input limit | 400 tokens, including special tokens and any prefixes |
| Overlap target | 50 tokens within the same page and detected section |
| Document prefix | Empty; section labels are not prepended |
| Question prefix | `Represent this sentence for searching relevant passages: ` |
| Search | Normalized embeddings with FAISS `IndexFlatIP` (cosine similarity) |
| Default results | 5 passages |

Settings are stored in `config/settings.yaml`. With `runtime.device: auto`, model execution prefers a working CUDA GPU and falls back to CPU if unavailable or on supported CUDA resource failures. FAISS search remains on CPU.

The pinned model is already cached on Billy's device. On another device, download it once before chunking or retrieval, which load cached model/tokenizer files locally:

```powershell
& $projectPython -B -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5', revision='5c38ec7c405ec4b44b94cc5a9bb96e735b38267a', device='cpu')"
```

Put the book PDFs in a `books/` folder in the project root. They are not included in this repository.

## Running the pipeline

Run stages one at a time and review each output before continuing. The current corpus and index have already been built; to try a question, use only the retrieval command below.

```powershell
& $projectPython -B .\src\ingest.py          # PDF text layers -> data/extracted/
& $projectPython -B .\src\clean.py           # Cleaned pages -> data/cleaned/
& $projectPython -B .\src\chunk.py           # Token-based passages -> data/chunks/
& $projectPython -B .\src\prepare_corpus.py  # Filter/deduplicate -> data/embedding/
& $projectPython -B .\src\embed.py           # Vectors, FAISS index and passage store
```

Review reports in `evaluation/`, especially `ingestion_report.json`, `cleaning_report.json`, `chunking_report.json`, `corpus_preparation_report.json` and `corpus_exclusions.jsonl`. The corpus filter policy is `config/corpus_filters.json`.

Changes to source text, chunking, the filter policy or embedding input settings require rebuilding the affected downstream stages. Embedding validates prepared inputs and can reuse matching vector caches.

### Try retrieval

```powershell
& $projectPython -B .\src\retrieve.py 'What is a data hazard in an instruction pipeline?' --top-k 5
```

The terminal prints ranked passages, PDF page references, chunk IDs and similarity scores. Text ending in `...` is a shortened preview. Add `--json` to return full passage records:

```powershell
& $projectPython -B .\src\retrieve.py 'What is a data hazard in an instruction pipeline?' --top-k 5 --json
```

Retrieval checks the saved index and passage alignment before searching. Existing indexes without artifact fingerprints are checked against the prepared corpus, preparation report and per-book vector caches; keep those files together.

### Tests

```powershell
& $projectPython -B -m pytest .\tests -q -p no:cacheprovider
```

The 10 retrieval checks and 8 device-selection/fallback checks passed. Three example searches also ran successfully against the real index on the Quadro M2000M. Earlier ingestion tests had unresolved failures concerning watermark-only image pages, excluded-directory symlinks and protection against a report overwriting a source PDF. The full test suite has not been confirmed passing after the latest changes.

## Repository layout

```
README.md
requirements.txt
config/settings.yaml
config/corpus_filters.json
books/                      # PDFs, not committed
data/
  book_manifest.json
  evaluation_questions.json # Jedaiah's evaluation set, pending
  extracted/  cleaned/  chunks/  embedding/  # generated, not committed
  passages.jsonl            # indexed passages, not committed
artifacts/index/
  index.faiss  index_config.json  vectors/  # generated, not committed
src/
  ingest.py  clean.py  chunk.py  prepare_corpus.py  embed.py
  runtime_device.py  retrieve.py
  generate.py  app.py  evaluate.py          # implementation pending
tests/
evaluation/
  ingestion_report.json  cleaning_report.json  chunking_report.json
  corpus_preparation_report.json  corpus_exclusions.jsonl
  report.json                             # formal evaluation pending
```

## Generated files and Git

Keep source code, configuration, tests and documentation in Git. Books, generated text, prepared passages, vector caches, FAISS indexes and model weights are excluded by `.gitignore`. Keep local copies needed to run retrieval; cloning the repository alone does not provide these generated artifacts or the cached model.

## Known limitations

- Extraction reads existing text layers only. Figures, diagrams and unencoded equations are not captured, and no OCR is used.
- Pages flagged for corrupted text, fragmented word spacing or insufficient image-page text are excluded from chunking. Useful content on those pages may therefore be unavailable.
- Cleaning and section detection use heuristics; residual extraction errors can remain. Review the reports and compare passages with their PDF pages.
- Each chunk stays within one PDF page and detected section. Relevant explanations can span several chunks or pages.
- Similarity search always returns the nearest passages, even for unsupported questions. Insufficient-evidence behaviour belongs to the pending generation stage.
- Answer generation, citation validation for generated answers, the Streamlit app and formal evaluation are still pending.
