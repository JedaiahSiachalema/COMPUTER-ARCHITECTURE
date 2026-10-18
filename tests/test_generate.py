
import unittest
from src.generate import generate_answer


PASSAGE = {
    "group_id": "G090",
    "subject": "Computer Architecture",
    "book_id": "G090_BOOK_003",
    "title": "Example textbook",
    "edition": "1",
    "page": 143,
    "chunk_id": "G090_BOOK_003_P0143_C02",
    "text": (
        "A cache is a small, fast memory that stores copies "
        "of frequently used data. The processor can access "
        "cached data more quickly than main memory."
    ),
}


class GenerationTests(unittest.TestCase):

    def test_supported_question(self):
        result = generate_answer(
            "What is a cache memory?", [PASSAGE]
        )

        self.assertFalse(result["insufficient_evidence"])
        self.assertTrue(result["answer"])
        self.assertEqual(result["sources"][0]["page"], 143)

    def test_empty_passages(self):
        result = generate_answer("Explain pipelining", [])

        self.assertTrue(result["insufficient_evidence"])
        self.assertEqual(result["sources"], [])

    def test_unrelated_question(self):
        result = generate_answer(
            "How does photosynthesis work?", [PASSAGE]
        )

        self.assertTrue(result["insufficient_evidence"])

    def test_missing_citation_metadata(self):
        passage = {"text": "Cache memory stores frequently used data."}

        result = generate_answer(
            "What is cache memory?", [passage]
        )

        self.assertTrue(result["insufficient_evidence"])
        self.assertEqual(result["sources"], [])

    def test_required_response_fields(self):
        result = generate_answer("What is a cache?", [PASSAGE])

        self.assertEqual(
            set(result),
            {"answer", "sources", "insufficient_evidence"},
        )


if __name__ == "__main__":
    unittest.main()
