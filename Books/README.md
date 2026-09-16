# 📚 Books

A personal collection of structured notes (and occasional tiny implementations) built while working through ML/AI books, chapter by chapter.

Each book gets its own folder here. Notes are written to be **concise and quick to revisit** — not a full transcript of the book, but a clean, structured pass with formulas, tables, and small examples added where they help understanding.

---

## Structure

```
books/
├── README.md                          ← you are here
├── <book-slug>/
│   ├── README.md                      ← (optional) book-level overview / table of contents
│   ├── chapter-01-<topic>/
│   │   ├── notes.md
│   │   └── implementation/            ← (optional) small code demos for concepts in this chapter
│   ├── chapter-02-<topic>/
│   │   └── notes.md
│   └── ...
└── <another-book-slug>/
    └── ...
```

- **`<book-slug>`** — lowercase, hyphenated short name of the book (e.g. `ai-engineering`).
- **`chapter-NN-<topic>`** — zero-padded chapter number + short topic slug, so folders sort correctly and are identifiable at a glance (e.g. `chapter-06-rag-and-agents`).
- **`notes.md`** — the structured notes for that chapter: headings per topic, tables for comparisons, LaTeX (`$...$` / `$$...$$`) for formulas, and short examples where they clarify a concept.
- **`implementation/`** — only present where a chapter's concepts were worth coding up (e.g. a tiny TF-IDF scorer, a toy LoRA implementation, an RRF ranking demo). Not every chapter has one.

---

## Notes conventions

- Formulas use GitHub-flavored math syntax (`$...$` inline, `$$...$$` block) — renders correctly in GitHub's file preview (not in raw file view).
- Tables are preferred over long prose for anything comparative (metrics, techniques, trade-offs).
- Examples are marked with `>` blockquotes so they're visually separable from core definitions.
- Notes are living documents — later chapters may get revised as understanding deepens or as a book gets re-read.

---
