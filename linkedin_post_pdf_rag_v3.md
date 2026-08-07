# LinkedIn Content Package — PDF-RAG (Compact Edition)

---

## Post Title

"Chat With Any PDF, Locally, For Free"

---

## Main Post

---

I built a CLI tool that lets you ask questions about any PDF and get cited, page-accurate answers — with no cloud uploads and no embedding costs.

It works in two steps. First, the PDF is chunked and embedded locally using HuggingFace sentence-transformers, then stored in ChromaDB on your machine. When you ask a question, the top matching chunks are retrieved and sent to Llama 3.3 70B via Groq, which answers strictly from the document and cites the page number.

If the answer is not in the PDF, it says so.

**Stack:** Python, LangChain, ChromaDB, sentence-transformers, Groq API, PyPDF

The biggest lesson was that chunking strategy matters more than model choice. A 200-character overlap between chunks improved retrieval quality more than any model swap did.

GitHub: [Your Link Here]

What chunking approach do you use in your RAG pipelines?

---

#RAG #LangChain #LLM #Python #OpenSource #AIEngineering #VectorDatabase #GenerativeAI #BuildInPublic #SoftwareEngineering #HuggingFace #ChromaDB #MachineLearning

---
---

## Short Version

---

Built a local RAG pipeline to query PDFs with exact page citations.

Stack: Python, LangChain, ChromaDB, HuggingFace embeddings, Llama 3.3 70B on Groq.

No embedding costs. Nothing leaves your machine during ingestion. The model cites its sources and declines to guess when the answer is not in the document.

GitHub: [Your Link Here]

#RAG #LangChain #Python #LLM #OpenSource

---
---

## Engagement Comment

---

The detail that made the biggest difference: using a 200-character overlap between chunks in the text splitter.

Without it, context gets cut mid-sentence at chunk boundaries and retrieval quality drops significantly. It is a small config change with a large impact on answer accuracy.

Happy to answer questions on the architecture.

---
---

## 5 Alternative Hooks

---

1. I stopped reading PDFs manually. Here is what I built instead.

2. Most PDF tools upload your data to a third-party server. Mine does not.

3. Llama 3.3 70B, ChromaDB, and zero embedding cost — here is the full stack.

4. There is no good reason to ctrl+F through a 200-page document anymore.

5. The best way to understand RAG is to build one end to end. Here is mine.

---
