"""Generate answers grounded in retrieved book passages.

TODO: Return answer, sources, and insufficient_evidence. Validate every
source identifier and acknowledge insufficient or conflicting evidence.
"""

"""Offline answer generation for Group 90."""
import re


def tokens(text):
    stopwords = {
        "the", "a", "an", "is", "are", "was", "what", "how",
        "why", "when", "where", "and", "or", "of", "to",
        "in", "for", "with", "computer", "architecture",
    }
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if len(w) > 1 and w not in stopwords}


def validate_sources(sources, passages):
    valid = {
        (str(p["book_id"]), int(p["page"]), str(p["chunk_id"]))
        for p in passages
        if p.get("book_id") and p.get("page") is not None
        and p.get("chunk_id")
    }

    result = []
    seen = set()

    for source in sources:
        try:
            key = (
                str(source["book_id"]),
                int(source["page"]),
                str(source["chunk_id"]),
            )
        except (KeyError, TypeError, ValueError):
            continue

        if key in valid and key not in seen:
            result.append({
                "book_id": key[0],
                "page": key[1],
                "chunk_id": key[2],
            })
            seen.add(key)

    return result


def generate_answer(question, passages):
    """Return the required Group 90 answer dictionary."""

    abstention = {
        "answer": (
            "I could not find enough evidence in the retrieved "
            "textbook passages to answer this question."
        ),
        "sources": [],
        "insufficient_evidence": True,
    }

    if not question.strip() or not passages:
        return abstention

    question_tokens = tokens(question)
    candidates = []

    for passage in passages:
        text = str(passage.get("text", "")).strip()
        if not text:
            continue

        sentences = re.split(r"(?<=[.!?])\s+|\n+", text)

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            overlap = len(question_tokens & tokens(sentence))

            if overlap:
                candidates.append(
                    (overlap, passage, sentence)
                )

    if not candidates:
        return abstention

    candidates.sort(key=lambda item: item[0], reverse=True)

    answer_sentences = []
    supporting_passages = []
    seen_sentences = set()

    for _, passage, sentence in candidates:
        normalized = sentence.lower()

        if normalized in seen_sentences:
            continue

        seen_sentences.add(normalized)
        answer_sentences.append(sentence)
        supporting_passages.append(passage)

        if len(answer_sentences) == 5:
            break

    sources = []

    for passage in supporting_passages:
        if all(passage.get(k) is not None for k in (
            "book_id", "page", "chunk_id"
        )):
            sources.append({
                "book_id": passage["book_id"],
                "page": passage["page"],
                "chunk_id": passage["chunk_id"],
            })

    sources = validate_sources(sources, passages)

    if not sources:
        return abstention

    return {
        "answer": " ".join(answer_sentences),
        "sources": sources,
        "insufficient_evidence": False,
    }
