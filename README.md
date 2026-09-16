# 🧠 AI/ML Learning Journal

A personal repository documenting my journey through AI/ML — structured notes from books I'm working through, small implementations to cement key concepts, and (eventually) full projects built along the way.

This isn't meant to be a polished course or tutorial for others — it's my own working notebook, kept clean enough that future-me (or anyone curious) can actually navigate it.

---

## 📁 Repository Structure

```
.
├── README.md                          ← you are here
├── books/
│   ├── README.md                      ← conventions for notes & implementations
│   ├── <book-slug>/
│   │   ├── chapter-01-<topic>/
│   │   │   ├── notes.md
│   │   │   └── *.ipynb                ← small implementations for concepts in this chapter
│   │   └── ...
│   └── <another-book-slug>/
│       └── ...
└── projects/                          ← (coming soon)
    └── <project-name>/
        └── ...
```

### `books/`
Structured, revision-friendly notes taken while reading ML/AI books — organized **one folder per book**, and **one folder per chapter** within each book. Some chapters also include small Jupyter notebooks implementing a concept from that chapter (e.g., a toy tokenizer, a Word2Vec-based recommender). See [`books/README.md`](./books/README.md) for the full folder/notes conventions.

**Books:**

| Book | Status |
|---|---|
| *AI Engineering: Building Applications with Foundation Models* — Chip Huyen | ✅ Completed |
| *Hands-On Large Language Models* — Jay Alammar & Maarten Grootendorst | 📖 In progress |

### `projects/`
Larger, standalone builds that go beyond book notes — applying these concepts to actual working systems (e.g., agentic pipelines, RAG applications, finetuning experiments). This folder is just getting started; more to come as projects are built.

---

## 🎯 Purpose

- Keep notes **concise and revision-friendly** — structured with headings, tables for comparisons, and formulas — rather than a full transcript of the source material.
- Pair theory with **small, hands-on implementations** wherever a concept benefits from actually seeing it run.
- Build up toward **real projects** that combine concepts across books and chapters (RAG, agents, finetuning, evaluation pipelines, etc.).

---

## 🛠️ Notes Conventions

- Math is written in GitHub-flavored LaTeX (`$...$` inline, `$$...$$` block) — renders in GitHub's file preview, not in raw file view.
- Comparative content (techniques, trade-offs, metrics) is presented in **tables** over long prose.
- Examples are called out with `>` blockquotes to stay visually distinct from core definitions.
- Notes are living documents and get revised as understanding deepens or books get revisited.

---

## 🙏 Acknowledgements / Sources

The notes in this repository are my own paraphrased summaries, written while working through the following books — all credit for the underlying concepts, frameworks, and explanations goes to their original authors. Nothing here is a substitute for reading the source material, which I'd genuinely recommend.

- **Chip Huyen** — *AI Engineering: Building Applications with Foundation Models*
- **Jay Alammar & Maarten Grootendorst** — *Hands-On Large Language Models*

If any note here misrepresents a concept from these books, that's a gap in my understanding, not the original material — feel free to point it out.
