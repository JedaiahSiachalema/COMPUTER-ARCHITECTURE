import json, pathlib
p = pathlib.Path("data/book_manifest.json")
books = json.loads(p.read_text(encoding="utf-8"))
for b in books:
    b["book_id"] = b["book_id"].replace("CA_BOOK", "G090_BOOK")
    b["group_id"] = "G090"
    if "related_book_ids" in b:
        b["related_book_ids"] = [x.replace("CA_BOOK", "G090_BOOK") for x in b["related_book_ids"]]
p.write_text(json.dumps(books, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")