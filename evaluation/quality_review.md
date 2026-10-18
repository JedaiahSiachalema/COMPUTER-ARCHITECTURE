# Cleaning and chunking quality review

Review date: 2026-10-18. Scope: review of the existing outputs, without changing scripts, rebuilding datasets, downloading a model, creating an environment, or running OCR.

## Decision

The pipeline produces structurally consistent records, but the current corpus needs quality corrections before embedding. The most important problems are damaged technical content, insufficient filtering of flagged passages, unreliable section labels, and inconsistent overlap. A stronger embedding model cannot restore missing mathematical meaning or scrambled reading order.

## Embedding recommendation

Use `BAAI/bge-small-en-v1.5` as the proposed local baseline, subject to the coordinator's common-model requirement for the integrated index. It is an English retrieval model with 384-dimensional vectors, a 512-token input limit, and an MIT license. Its size makes it a reasonable candidate for this laptop's 16 GB RAM and 4 GB GPU; speed and actual compatibility with this model have not yet been tested.

Official source: <https://huggingface.co/BAAI/bge-small-en-v1.5>

Pin the exact model revision. For short questions, the model card recommends the query prefix `Represent this sentence for searching relevant passages: `, with no such instruction added to document passages. Use normalized embeddings with the existing cosine/inner-product index design. This recommendation has not been written into the configuration.

Proposed starting chunk target: 350-400 tokens including any section prefix, with approximately 50 tokens of overlap where context continues. Verify all final embedding inputs, including special tokens, fit within 512 tokens. Tune these choices using retrieval evaluation rather than assuming this target is optimal. No tokenizer audit was performed during this review, so the existing chunks' truncation rate is unknown.

## Evidence and scope

- Parsed the existing extracted, cleaned, and chunk files for all 18 books.
- Reconstructed chunk groups in memory using the existing pure chunking functions and verified the resulting chunk text against all 27,499 stored chunks. This did not execute the pipeline's writing functions.
- Checked chunk ID uniqueness, book/group/subject/title consistency, and page-range validity against the manifest and page records; no errors were found in these checks.
- Inspected a midpoint text sample from each book, front matter samples, flagged passages, section-boundary examples, and raw-versus-cleaned technical text.
- Visually compared three complete source PDF pages: BOOK_008 PDF positions 74 and 524, and BOOK_018 PDF position 358. PDF positions differ from the page numbers printed on these pages.
- This is an automated corpus audit with selected manual inspection, not a claim that every passage was read or that retrieval accuracy has been measured. The ingestion test suite was not rerun; no cleaning/chunking test files currently exist.

## Measured results

| Check | Result |
|---|---:|
| Books | 18 |
| Extracted/cleaned page records | 14,237 each |
| Pages marked usable | 14,020 |
| Pages marked unusable | 217 |
| Pages flagged for fragmented spacing | 157 |
| Chunks | 27,499 |
| Chunks containing flagged spacing pages | 330 |
| Minimum / median / maximum chunk characters | 71 / 1,341 / 2,200 |
| 95th percentile chunk characters | 1,899 |
| Chunks above 1,800 characters | 1,755 |
| Adjacent chunk boundaries within books | 27,481 |
| Boundaries with no carried-over units | 15,518 (56.5%) |
| Boundaries with carried-over units | 11,963 |
| Largest actual carried-over text, including separators | 226 characters |
| Chunks spanning multiple PDF pages | 10,491 (38.2%) |
| Largest start-to-end page range | 9 PDF positions |
| Chunks containing more than one unit section label | 6,862 (25.0%) |
| Chunks with no section label | 105 |
| Chunks carrying an obvious prose-style chapter label | 56 |
| Exact duplicate text groups across books | 825 pairs |
| Raw pages containing private-use/replacement characters | 26 |
| Private-use/replacement characters in those raw pages | 286 |

Mixed section labels include parent/subsection changes, running headers, and contents entries. This count is a warning indicator, not proof that all 6,862 chunks combine unrelated material. Overlap was measured by matching the actual suffix/prefix units carried between adjacent groups, rather than counting coincidental repeated words.

All 825 exact duplicate pairs are between BOOK_008 (ARM) and BOOK_009 (RISC-V). This is understandable for closely related editions but can reduce the diversity of top-k retrieval results. Both source citations should be retained if duplicate passage text is consolidated.

The cleaning report records removal of 10,619 running-header lines and 522 page-number lines. Raw extraction remains available separately, which makes corrections possible without OCR.

## Findings, in priority order

### 1. Technical content is not reliably preserved

`src/clean.py:47` and `src/clean.py:54` delete all private-use glyphs and replacement characters. Some private-use glyphs are mathematical bracket pieces. Deleting them silently can remove meaningful notation. Extraction already loses other symbols and reading order before cleaning, so this problem cannot be solved solely by changing the cleaner.

Examples:

- BOOK_008, PDF position 74 (printed page 49): the source page shows a ratio of two sums and Amdahl's Law with fraction bars, equals signs, and addition/subtraction. The extracted text is already incomplete; cleaned text contains equation fragments and mixes the right-hand margin explanation into the main body. Its raw `unicode_review` flag remains on the cleaned page, but chunk records do not carry this flag.
- BOOK_008, PDF position 524 (printed page 404): the source shows powers such as `2^10` and floor brackets for block-address mapping. The extracted/cleaned text renders some powers as `210` and removes bracket glyphs. That changes mathematical meaning.
- BOOK_018, PDF position 358 (printed page 339): source C code is readable visually, but the text layer has a malformed `printf` expression and displaced comments/braces. Cleaning preserves these defects. This page is flagged for fragmented spacing and still enters the chunk corpus.

Recommended correction: preserve diagnostic flags through to passages and exclude unresolved damaged passages from the embedding input. Protect code, equations, and tables from generic prose cleanup. Use layout-aware extraction from the existing text layer where it succeeds. If a passage would require OCR, exclude it under the user's instruction; do not infer missing operators or invent corrected code.

### 2. Flagging does not prevent low-quality material entering the corpus

`src/clean.py:126` sets usability solely from a minimum alphanumeric count. `src/clean.py:127` records fragmented spacing without repairing or excluding it. `src/chunk.py:79` and `src/chunk.py:145` propagate the spacing flag but still include the text. Consequently, 330 chunks contain material from the 157 flagged pages.

The spacing flag also catches some legitimate diagrams and code, so blanket automatic word joining would be unsafe. Flagged content needs review or exclusion, with a reason recorded. The current minimum-length rule is not a semantic quality check.

### 3. Chunk sizes are measured in characters, not the model's tokens

`src/chunk.py:26` through `src/chunk.py:28` use 1,500 target characters, 2,200 maximum characters, and 200 overlap characters. The four-characters-per-token assumption does not guarantee that technical terms, assembly, bit strings, or damaged text fit the embedding model.

`src/embed.py:96` samples only 400 chunks for truncation, and `src/embed.py:57` can prepend a section label. The full final text sent to the encoder needs a tokenizer-based limit check before encoding. No exact number of over-limit chunks can be claimed until the selected model tokenizer is available.

### 4. Overlap is usually absent

`src/chunk.py:85` carries only complete units whose text fits the 200-character budget. If the last paragraph/unit is longer than that budget, no tail is retained. This explains the 15,518 boundaries with zero overlap. Paragraph separators are not counted in the overlap budget, which also explains observed overlaps above 200 characters.

Recommended correction: use token-aware overlap within an appropriate continuing section; do not carry unrelated context across a chapter or other hard boundary. Inspect examples where an explanation or worked example crosses a chunk boundary.

### 5. Section labels and boundaries are unreliable

`src/chunk.py:31` accepts ordinary sentences beginning with `Chapter <number>` as headings. For example, `G090_BOOK_001_P0013_C01` is labelled with the prose sentence beginning `Chapter 3 is about performance...`. Contents entries and surviving running headers are also treated as headings.

`src/chunk.py:65` through `src/chunk.py:81` can assign a heading found later inside a paragraph to the whole paragraph. Grouping does not flush on section changes, and the final chunk receives only the first unit's label (`src/chunk.py:142`). A label can therefore be inaccurate for part of the passage. Section text is included in embeddings by default, so inaccurate labels would affect retrieval.

Examples: `G090_BOOK_001_P0006_C01` combines dedication/contents entries; `G090_BOOK_001_P0054_C02` includes labels for both mesh/network material and the following performance section/header. These are reasons to review boundaries, not reasons to reject every parent/subsection combination.

Recommended correction: distinguish document headings from prose, contents entries, and page furniture; preserve a reliable heading hierarchy and split on major section transitions. Disable section prepending until labels are reliable.

### 6. Page attribution needs finer provenance

10,491 chunks cross PDF pages. Records retain `page` and `page_end`, but no offsets or per-fragment source-page map. A consumer returning only the required `page` field could cite the first page even when the supporting statement appears on the second page.

Example: `G090_BOOK_001_P0010_C02` spans PDF positions 10 to 12, joining an index/contents tail to preface text while skipping an unusable page. Record actual contributing page spans/offsets, or use single-page chunks where precise attribution cannot otherwise be guaranteed. Start/end ranges alone are too coarse to locate individual evidence statements reliably.

### 7. Noninstructional material and duplicates remain

The first chunks of several books contain publisher advertisements, endorsements, copyright details, dedications, or contents listings. Verified examples are `G090_BOOK_001_P0002_C01`, which starts with the title and Wiley series advertising, and `G090_BOOK_018_P0002_C01`, which starts with publisher/copyright material. Both span PDF positions 2-3.

Filter clearly noninstructional front/back matter from the retrieval corpus while keeping useful learning material, worked examples, relevant prefaces, and references when justified. Avoid blanket deletion of all first pages or all numeric/diagram-heavy pages. Consolidate exact duplicate text or diversify retrieval results without losing alternative source provenance.

### 8. Configuration must be aligned before embedding

`config/settings.yaml:19` uses `embedding.model_id`, while `src/embed.py:33` reads `embedding.model_name`. The revision is unset and chunk token settings are unset. Downloading the model alone will not resolve this mismatch. Align the keys and pin the coordinator-approved model/revision when the embedding setup step is authorized.

## Suggested order for the next work

1. Correct the cleaning quality gate and preserve technical-content diagnostics; exclude unresolved content requiring OCR.
2. Verify a small set of prose, code, equation, table, and sidebar examples against their source text layers and rendered pages.
3. Correct heading detection, token-based chunking, overlap, and source-page mapping.
4. Rebuild only the affected cleaning/chunk outputs, then repeat the quality checks and inspect representative samples.
5. Configure the selected embedding model and exact revision, check all token lengths, and build the index.
6. Measure retrieval on held-out Computer Architecture questions before drawing conclusions about model quality.

Only this review document and temporary PDF page previews were created during the review. No source scripts, manifest entries, cleaned/chunk data, settings, environments, or indexes were changed.
