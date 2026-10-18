"""Common retrieval interface for the Computer Architecture chatbot."""


def retrieve(question: str, top_k: int = 5) -> list:
    """Return passages with text, score, book_id, title, page, and chunk_id."""
    raise NotImplementedError("Retrieval is pending implementation.")
