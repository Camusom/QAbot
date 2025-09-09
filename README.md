RAG-based PDF QA Chatbot (LangChain + Chroma + Gradio + Ollama)

Overview:
- Upload a PDF and ask questions about it.
- Local RAG pipeline: Chroma vector store, optional cross‑encoder reranker, citations, persistent cache.
- Controls in the UI: top_k, chunk size, chunk overlap, reranker on/off and k_final, LLM choice (Ollama), embedding model (HF or Ollama).
- Safety behavior: answers only from supplied context; says “I don’t know” when the answer isn’t present.

Main components:
- Document loader: PyPDFLoader
- Text splitting: RecursiveCharacterTextSplitter
- Embeddings: HF sentence-transformers or Ollama embeddings
- Vector DB: Chroma with on‑disk persistence per document/settings
- Retriever: top_k from Chroma; optional rerank to k_final via cross-encoder
- LLMs: mixtral, mistral, llama3:8b, llama3:8b-instruct-q4_0 via Ollama
- UI: Gradio


Prerequisites
- OS: macOS, Linux, or Windows
- Python: 3.10+
- If using local models: Ollama installed and running on http://localhost:11434
- Internet access the first time to download models (Ollama or HF)


Python dependencies:
- langchain
- langchain-community
- chromadb
- gradio
- pypdf
- sentence-transformers
- transformers
- accelerate
- requests
- Quick start

To Use:
Clone repo and create a virtual environment
- python -m venv .venv
- source .venv/bin/activate (Windows: .venv\Scripts\activate)
- pip install -r requirements.txt
If you will use any Ollama LLM or embedding
- Install Ollama: https://ollama.com/download
- Start service: ollama serve (some platforms autostart)
- Verify: curl http://localhost:11434/api/tags
Pre‑pull models (recommended)
- ollama pull mistral
- ollama pull mixtral
- ollama pull llama3:8b
- ollama pull llama3:8b-instruct-q4_0
- ollama pull nomic-embed-text
- ollama pull mxbai-embed-large
Run
- python app.py
- Open http://localhost:7860
Upload a PDF, choose models and retrieval settings, ask questions.


Hardware guidance (rules of thumb)
Note: exact usage varies by quantization/builds. These are conservative local CPU numbers; any capable GPU reduces latency.

llama3:8b
- With q4_0 quantization, disk ~4.5–5.5 GB.
- 16 GB system RAM recommended; workable on 8–12 GB with swap but slower.
- GPU acceleration comfortable at 8–12 GB VRAM.
llama3:8b-instruct-q4_0
- Smallest footprint of the Llama3 options; good for 8–16 GB RAM machines.
mistral (7B)
- Similar to llama3:8b q4_0. 16 GB RAM recommended for smooth CPU; 8–12 GB VRAM preferred for GPU.
mixtral (8x7B MoE)
- Heavy. Plan 32–64 GB system RAM or 24–48 GB VRAM. If resources are tight, avoid this locally.
phi3:mini
- Smallest model available; should work on most CPUs. Choose this not enough RAM error persists.




Embeddings
HF: sentence-transformers/all-MiniLM-L6-v2
- Lightweight (~100–200 MB). CPU‑friendly, works on most machines.
Ollama: nomic-embed-text
- Small; fine on CPU.
Ollama: mxbai-embed-large
- Larger than the above but still lightweight compared to LLMs; CPU is usually fine.


If you see out‑of‑memory or very slow inference:
- Switch to llama3:8b-instruct-q4_0, mistral, or phi3:mini (smallest model).
- Lower Retrieval top_k and/or k_final.
- Reduce chunk size.
- Close memory‑hungry apps.


How retrieval tuning affects accuracy and latency:
Chunk size
- Small chunks (300–600 chars): better pinpointing, more chunks stored, faster embed/build, but may lose cross‑sentence context and increase retrieval steps.
- Medium chunks (800–1200): good default for reports and PDFs; balances context and precision.
- Large chunks (1300–1500+): fewer, bigger chunks; can improve recall for long passages but increases token usage and may add noise.

Chunk overlap
- 10–20% of chunk size is a good default. Avoid 0 for narrative text; use minimal overlap for structured tables.
- Higher overlap increases index size and retrieval time but reduces boundary cuts.

Top‑K (retrieval candidates)
- Higher improves recall; cost grows roughly linearly in retrieval + reranking time.
- Typical range: 5–10.

Reranker (cross‑encoder)
- Greatly improves precision at the top; adds latency proportional to k_candidates.

Remember: Use k_final smaller than top_k (e.g., top_k=8, k_final=3 or 4). Top_k extracts the top choices and k_final chooses among the specified k best of those top_k choices. 


How the reranker works:
- The base retriever returns top_k chunks using vector similarity.
- A cross‑encoder (ms-marco-MiniLM-L-6-v2) scores query–chunk pairs with a small transformer.
- The ContextualCompressionRetriever keeps only the top k_final scored chunks.
- Tradeoff: better answer grounding and fewer hallucinations at the cost of extra compute.


How the cache and persistence work:
- A unique key is built from: file path + file modified time + embedding model + chunk size + chunk overlap.
For each key:
- If a persisted Chroma directory exists (chroma_store/), it reloads it.
- Otherwise the PDF is loaded, split, embedded, and stored; then persisted to disk.
- An in‑memory dict (index_cache) holds the opened Chroma instance during the session.
To force a rebuild:
- change the PDF (mtime), change settings, or delete the corresponding folder under chroma_store.
To clear all caches:
- delete the chroma_store directory.


Citations:
- The app returns page hints like “Sources: 1.page5, 2.page12”.
- Page numbers are 1‑based for readability.


Common issues and fixes:
“Ollama server not reachable”: 
- start Ollama (ollama serve) or install it.
Model download is slow:
- pre‑pull models before running the app.
CUDA not used:
- Ollama and sentence-transformers can run CPU‑only; GPU acceleration depends on your local setup and drivers.
Mixed environments:
- use a virtual environment; upgrade pip; reinstall requirements if versions conflict.


Security and privacy:
- All inference runs locally; your PDFs are not sent to external APIs.
- Chroma persistence stores embeddings on disk under chroma_store; delete this folder to remove stored vectors.


License and data:
- Ensure the PDFs you test with are yours to process.
- Check model licenses (Ollama model cards and HF model cards) before redistribution.
