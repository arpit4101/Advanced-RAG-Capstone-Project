# 🧠 Enterprise-Grade Advanced RAG Assistant

[![Python Version](https://img.shields.io/badge/Python-3.12%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-v0.3.0%2B-green?logo=langchain&logoColor=white)](https://www.langchain.com/)
[![Vector DB](https://img.shields.io/badge/Vector_DB-FAISS%20%7C%20Pinecone-orange?logo=database&logoColor=white)](https://github.com/facebookresearch/faiss)
[![UI Framework](https://img.shields.io/badge/Streamlit-1.50%2B-red?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Cross-Encoder](https://img.shields.io/badge/Cross--Encoder-MS_MARCO-purple?logo=huggingface&logoColor=white)](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2)
[![OpenAI](https://img.shields.io/badge/LLM-GPT--4o--Mini-black?logo=openai&logoColor=white)](https://openai.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **A production-grade Retrieval-Augmented Generation (RAG) system engineered to overcome standard naive RAG limitations through multi-query expansion, hybrid dense-sparse retrieval, hierarchical parent-child chunk recovery, cross-encoder neural reranking, and dynamic context refinement.**

---

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Why Naive RAG Fails](#-why-naive-rag-fails)
- [High-Level Architecture](#-high-level-architecture)
- [Advanced RAG Techniques Implemented](#-advanced-rag-techniques-implemented)
- [Core Components & Modular Design](#-core-components--modular-design)
- [System Lifecycle & Data Flow](#-system-lifecycle--data-flow)
- [Repository Structure](#-repository-structure)
- [Tech Stack](#-tech-stack)
- [Getting Started & Setup](#-getting-started--setup)
- [Interactive UI & Retrieval Debug Mode](#-interactive-ui--retrieval-debug-mode)
- [Enterprise Scaling & Future Enhancements](#-enterprise-scaling--future-enhancements)
- [License](#-license)

---

## 🎯 Project Overview

In standard enterprise environments, knowledge retrieval powered by naive RAG architectures suffers from severe real-world failure modes: semantic mismatch on domain terminology, inability to locate exact IDs or acronyms, loss of critical surrounding context during text chunking, and hallucinated answers triggered by low-relevance passages.

This repository implements a **multi-stage, state-of-the-art Advanced RAG pipeline** designed to provide verifiable, audit-ready answers from heterogeneous corporate repositories (`.pdf`, `.docx`, `.txt`). It bridges the gap between semantic understanding and lexical precision through **Hybrid Search (FAISS / Pinecone + BM25)**, solves the chunk-size trade-off using **Hierarchical Parent-Child Indexing**, mitigates false positives via **Neural Cross-Encoder Reranking**, and maximizes LLM reasoning quality through **Dynamic Context Budgeting & Formatting**.

The entire system is exposed via a full-featured **Streamlit Web Application** featuring white-box retrieval inspection ("Retrieval Debug Mode"), dynamic document ingestion, index lifecycle operations, and sliding-window conversational memory.

---

## ⚠️ Why Naive RAG Fails

| Naive RAG Vulnerability | Manifestation in Production | How This Architecture Solves It |
| :--- | :--- | :--- |
| **Semantic / Lexical Divergence** | Users search with synonyms or colloquialisms ("time off"), while policies use legal terms ("annual leave"). Single vector queries miss target passages. | **Multi-Query Expansion**: Generates multiple diverse semantic query formulations while disambiguating conversational references. |
| **Out-of-Vocabulary (OOV) & Acronym Blindness** | Dense embeddings struggle with exact codes, software versions, and policy identifiers (e.g., `SOC 2`, `v2.4`, `TN-4410`). | **Hybrid Search with Reciprocal Rank Fusion (RRF)**: Merges dense vector embeddings with sparse BM25 token matching. |
| **The Chunk-Size Dilemma** | Small chunks embed accurately but lack synthesis context; large chunks preserve context but dilute vector similarity. | **Hierarchical Parent-Child Retrieval**: Searches high-resolution child chunks (400 chars) and maps back to broad parent chunks (1600 chars) for generation. |
| **Top-K Noise & Distractor Chunks** | Dense retrievers always return top-$K$ candidates, even when documents are irrelevant, triggering LLM hallucinations. | **Cross-Encoder Reranking & Score Thresholding**: Scores candidate text pairs jointly with bidirectional cross-attention, filtering out weak matches (`MIN_RERANK_SCORE`). |
| **"Lost-in-the-Middle" Phenomenon** | LLMs degrade in reasoning accuracy when bloated, irrelevant context fills the prompt window. | **Context Refinement & Budget Enforcement**: Enforces a strict character budget (`MAX_CONTEXT_CHARS`) and structures cited text blocks. |

---

## 🏛️ High-Level Architecture

The system decouples **Offline Data Ingestion & Indexing** from **Online Query Expansion, Hybrid Retrieval, Reranking, and Generation**.

```mermaid
flowchart TD
    subgraph INGESTION["1. Document Ingestion & Hierarchical Indexing"]
        A["Source Documents (.pdf, .docx, .txt)"] --> B["Multi-Format Document Loaders"]
        B --> C["Regex Text Cleaning & Metadata Extraction"]
        C --> D["Parent-Child Recursive Character Splitting"]
        D -->|Parent Chunks 1600 chars| E[("Persistent Document Store: doc_store.json")]
        D -->|Child Chunks 400 chars| F["OpenAI text-embedding-3-small"]
        F --> G[("Dense Vector Store: FAISS / Pinecone")]
        E -.->|Build In-Memory| H["Sparse BM25 Index"]
    end

    subgraph QUERY["2. Multi-Stage Retrieval & Synthesis"]
        Q["User Query & Chat History"] --> I["Multi-Query Expansion (GPT-4o-Mini)"]
        I -->|Original + N Query Variants| J["Hybrid Retrieval Engine"]
        
        G -->|Dense Similarity Search| J
        H -->|Sparse Keyword Search| J
        
        J --> K["Reciprocal Rank Fusion (RRF)"]
        K -->|Top Fused Child Chunks| L["Parent Context Recovery Engine"]
        E -->|O(1) Hash Map Lookup| L
        
        L -->|Parent Chunks| M["Cross-Encoder Reranker (ms-marco-MiniLM-L-6-v2)"]
        M --> N["Relevance Score Thresholding (MIN_RERANK_SCORE)"]
        N --> O["Context Refinement & Token Budget Manager"]
        
        O -->|Curated Context + Sliding History| P["LLM Synthesis Engine (GPT-4o-Mini)"]
        P --> RES["Grounded Answer + Verifiable Source Citations"]
    end
```

---

## 🚀 Advanced RAG Techniques Implemented

### 1. Hierarchical Parent-Child Chunking
- **Mechanism**: Documents are split into two decoupled layers:
  - **Parent Chunks** (`PARENT_CHUNK_SIZE = 1600`, `OVERLAP = 200`): Large semantic paragraphs containing the full rationale, clauses, and narrative necessary for the LLM to generate deep, accurate answers.
  - **Child Chunks** (`CHILD_CHUNK_SIZE = 400`, `OVERLAP = 60`): Fine-grained passages extracted from within each parent, carrying an explicit reference link (`parent_id = filename::parent_idx`).
- **Advantage**: Dense vector databases search high-specificity child vectors to maximize cosine similarity precision. Once retrieved, child chunks are discarded in favor of their corresponding parent documents via an $O(1)$ memory lookup, eliminating truncated context.

### 2. Multi-Query Expansion & Conversational Disambiguation
- **Mechanism**: The user's query is intercepted by an LLM prompt that produces $N$ diverse semantic reformulations (`NUM_QUERY_VARIANTS = 3`). When chat history is present, the prompt performs coreference resolution, rewriting ambiguous pronouns ("does it apply to remote employees?") into self-contained search queries ("does the TechNova equipment policy apply to remote employees?").
- **Advantage**: Bypasses the vocabulary mismatch problem between user phrasing and internal corporate documentation.

### 3. Dual-Mode Hybrid Search (Dense + Sparse)
- **Mechanism**: For every query variant generated:
  - **Dense Retrieval**: Calculates cosine similarity against OpenAI vector embeddings using local **FAISS** (`IndexFlatL2` / cosine) or managed cloud **Pinecone**.
  - **Sparse Retrieval**: Executes inverted-index token matching via **BM25** (`rank-bm25`).
- **Advantage**: Captures deep conceptual relationships while guaranteeing recall on exact alphanumeric tokens (ticket numbers, policy codes, acronyms).

### 4. Reciprocal Rank Fusion (RRF)
- **Mechanism**: Rather than attempting naive raw-score normalization between uncalibrated vector distances and unbounded BM25 scores, results are fused using position-based RRF:
  $$\text{RRF Score}(d) = \sum_{m \in M} \frac{1}{k_{\text{constant}} + \text{rank}_m(d)}$$
  *(where $k_{\text{constant}} = 60$, and $M$ represents the set of all ranked retrieval passes).*
- **Advantage**: Documents that consistently appear near the top of multiple search queries and across both search modalities accumulate the highest scores naturally, without score skew.

### 5. Cross-Encoder Neural Reranking (with Dynamic Fallback)
- **Mechanism**: Passes concatenated `(query, document)` text pairs through a transformer cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) that evaluates joint token-level cross-attention.
- **Resilience Architecture**: If PyTorch or CUDA/OpenMP dependencies are unavailable on resource-constrained host machines, the system automatically falls back to an internal **Normalized Keyword Overlap Reranker**, ensuring zero pipeline interruption.

### 6. Dynamic Context Budgeting & Threshold Filtering
- **Mechanism**:
  - **Threshold Filtering**: Any chunk scoring below `MIN_RERANK_SCORE = -4.0` is purged to prevent hallucination induction.
  - **Hard Context Budgeting**: Selected parent chunks are assembled under a hard ceiling (`MAX_CONTEXT_CHARS = 8000`). If a parent chunk exceeds the remaining allocation, it is rejected, preserving room for higher-priority content.
- **Advantage**: Prevents "Lost-in-the-Middle" degradation where LLMs miss critical needle-in-haystack details when submerged in excessively long prompts.

### 7. Sliding-Window Conversational Memory
- **Mechanism**: Managed through `InMemoryChatMessageHistory` with a bounded sliding window (`MEMORY_WINDOW = 4` exchanges, keeping 8 total messages).
- **Advantage**: Maintains continuity for multi-turn dialogue while strictly bounding token consumption and preventing prompt bloat.

---

## 🧩 Core Components & Modular Design

```
src/advanced_rag_capstone_project/
├── ingestion.py       # Format loaders (.pdf, .docx, .txt), cleaning regex, metadata enrichment
├── chunking.py        # RecursiveCharacterTextSplitter with hierarchical parent-child linking
├── indexing.py        # Vector store lifecycle (FAISS/Pinecone), document store JSON persistence
├── retrieval.py       # Multi-query generation, BM25 retriever, dense search, RRF result fusion
├── reranking.py       # Cross-encoder inference, caching, and graceful keyword fallback
├── refinement.py      # Relevance thresholding, token budget enforcement, citation formatting
├── memory.py          # Session chat history management and conversational formatting
└── rag_pipeline.py    # Master pipeline orchestrator connecting retrieval to LLM generation
```

- **`config.py`**: Centralized, environment-driven configuration module with strict type-casting, validation helpers, and fallback defaults.
- **`llm_client.py`**: Singleton-patterned, LRU-cached factory functions delivering thread-safe instances of `ChatOpenAI` and `OpenAIEmbeddings`.
- **`app.py`**: Production-ready Streamlit frontend providing document uploading, index management, interactive chat, and live telemetry inspection.

---

## 🔄 System Lifecycle & Data Flow

### Ingestion Lifecycle

```
[Raw Files: PDF/DOCX/TXT]
          │
          ▼
   load_document() ─────────► [Format Parsing & PyPDF/Docx2txt/TextLoader]
          │
          ▼
     clean_text() ──────────► [Hyphen De-breaking, Whitespace Normalization]
          │
          ▼
    add_metadata() ─────────► [Attach: filename, doc_type, page_number]
          │
          ▼
   split_documents() ───────► [Recursive Splitting into Parents (1600c) & Children (400c)]
          │
          ├───► save_doc_store() ──────► [Persist parents & children to doc_store.json]
          │
          └───► create_vector_store() ──► [Embed children via text-embedding-3-small -> FAISS / Pinecone]
```

### Generation Lifecycle

```
[User Question]
       │
       ▼
generate_query_variants() ──► [LLM produces original + 3 query expansions]
       │
       ▼
   hybrid_retrieve() ───────► [Run Dense Vector Search + BM25 Sparse Search per query]
       │
       ▼
     fuse_results() ────────► [Reciprocal Rank Fusion (RRF) -> Top candidate children]
       │
       ▼
   recover_parents() ───────► [Map chunk_id -> parent_id via doc_store in-memory lookup]
       │
       ▼
   rerank_documents() ──────► [Cross-Encoder ms-marco-MiniLM scores (query, parent) pairs]
       │
       ▼
   drop_irrelevant() ───────► [Filter out passages where score < MIN_RERANK_SCORE]
       │
       ▼
apply_context_budget() ─────► [Trim context to fit MAX_CONTEXT_CHARS ceiling]
       │
       ▼
   format_context() ────────► [Enrich passages with [1] Source: filename - Page]
       │
       ▼
   generate_answer() ───────► [ChatOpenAI gpt-4o-mini generates answer citing [1], [2]]
       │
       ▼
[Response to User + Citations + Full Step-by-Step Debug Telemetry]
```

---

## 📂 Repository Structure

```ascii
.
├── config.py                               # Central runtime configuration & environment loading
├── llm_client.py                           # Cached factories for ChatOpenAI & OpenAIEmbeddings
├── app.py                                  # Streamlit UI with document controls & debug inspector
├── pyproject.toml                          # Project metadata, build backend (uv), & dependencies
├── requirements.txt                        # Pip dependency manifest
├── env.example                             # Template environment variable configuration
├── projectflow.txt                         # Architectural implementation sequence guide
│
├── data/
│   ├── documents/                          # Source corporate documents for knowledge base
│   │   ├── technova_employee_handbook.pdf  # Comprehensive company policies (PDF)
│   │   ├── technova_product_guide.pdf      # Technical product specs (PDF)
│   │   ├── technova_security_policy.docx   # IT security protocols (DOCX)
│   │   └── technova_support_policy.txt     # SLA & support guidelines (Plain Text)
│   └── evaluation/                         # Benchmarking datasets (optional)
│
├── scripts/
│   ├── build_index.py                      # Standalone CLI indexing script
│   └── generate_sample_documents.py        # Synthetic test document generation utility
│
├── src/advanced_rag_capstone_project/      # Core RAG package
│   ├── __init__.py                         # Package initialization
│   ├── ingestion.py                        # Ingestion, cleaning regex, and metadata extraction
│   ├── chunking.py                         # Hierarchical parent-child text segmentation
│   ├── indexing.py                         # Storage management (FAISS, Pinecone, doc_store.json)
│   ├── retrieval.py                        # Multi-query rewrites, hybrid search, RRF fusion
│   ├── reranking.py                        # Cross-Encoder neural reranking with keyword fallback
│   ├── refinement.py                       # Relevance filtering, context budgeting, formatting
│   ├── memory.py                           # Sliding-window conversational chat memory
│   └── rag_pipeline.py                     # Master execution pipeline & answer generation
│
└── vectorstore/                            # Local storage artifacts (git-ignored in production)
    ├── doc_store.json                      # Serialized parents & children dictionary store
    └── faiss/                              # Serialized local FAISS index (index.faiss, index.pkl)
```

---

## 🛠️ Tech Stack

| Domain | Technology / Library | Specification / Model | Purpose |
| :--- | :--- | :--- | :--- |
| **Language & Runtime** | Python | `>= 3.12` | Modern typing, pattern matching, high-performance runtime |
| **Package Management** | `uv` / `pip` | Astral `uv_build` backend | Lightning-fast deterministic dependency resolution |
| **LLM Engine** | OpenAI API / LangChain | `gpt-4o-mini` (`temperature=0.0`) | Multi-query expansion & strictly grounded generation |
| **Embedding Model** | OpenAI Embeddings | `text-embedding-3-small` | 1536-dimensional semantic dense representations |
| **Dense Vector Store** | FAISS / Pinecone | `faiss-cpu >= 1.8.0` / `langchain-pinecone` | High-speed local indexing or scalable cloud vector DB |
| **Sparse Retrieval** | Rank-BM25 | `rank-bm25 >= 0.2.2` | Lexical frequency-inverse document frequency matching |
| **Neural Reranking** | Sentence-Transformers | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Transformer joint cross-attention relevance scoring |
| **Document Parsers** | PyPDF, Docx2txt | `pypdf >= 5.0.0`, `docx2txt >= 0.8` | Deep text extraction from structured binaries |
| **UI Framework** | Streamlit | `>= 1.50.0` | Reactive web dashboard with live telemetry debugging |

---

## 💻 Getting Started & Setup

### Prerequisites
- **Python 3.12+** installed on your machine.
- An **OpenAI API Key** with active credits.
- *(Optional)* A **Pinecone API Key** if running cloud vector storage.

---

### Step 1: Clone the Repository

```bash
git clone https://github.com/arpit4101/Advanced-RAG-Capstone-Project.git
cd Advanced-RAG-Capstone-Project
```

---

### Step 2: Set Up Virtual Environment

#### Option A: Using `uv` (Recommended — 10x faster)
```bash
# Create virtual environment with Python 3.12
uv venv --python 3.12
source .venv/bin/activate

# Install all dependencies including CrossEncoder & SentenceTransformers
uv sync
# Or install directly via requirements:
uv pip install -r requirements.txt
```

#### Option B: Using Standard Python `venv`
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

### Step 3: Configure Environment Variables

Copy the example configuration file:
```bash
cp env.example .env
```

Open `.env` in your editor and enter your credentials:
```dotenv
# Required
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Models (Defaults are pre-configured)
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Vector Database: "faiss" (local, default) or "pinecone" (cloud)
VECTOR_DB=faiss

# Required ONLY if VECTOR_DB=pinecone
PINECONE_API_KEY=
PINECONE_INDEX_NAME=advanced-rag
PINECONE_NAMESPACE=default

# Reranker selection: "cross-encoder" or "simple"
RERANKER=cross-encoder
CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

---

### Step 4: Ingest & Build the Vector Database

You can build the index directly via the command line or through the UI.

#### Option A: Command-Line Build
```bash
# Build the index over sample documents in data/documents
python scripts/build_index.py
```
*Expected terminal output:*
```
Reading documents from: data/documents
Index built successfully.
    Pages / documents loaded : 12
    Parent Chunks (Context)  : 24
    Child Chunks (Searched)  : 96
    Vector Database          : faiss

Now run: streamlit run app.py
```

#### Option B: Regenerate Fresh Sample Documents (Optional)
If you wish to test document generation from scratch:
```bash
python scripts/generate_sample_documents.py
python scripts/build_index.py
```

---

### Step 5: Launch the Streamlit Application

```bash
streamlit run app.py
```

> [!TIP]
> **macOS Notice (OpenMP Collision Fix):** If you encounter an OpenMP collision (`OMP: Error #15: Initializing libomp.dylib`), start the app with:
> ```bash
> KMP_DUPLICATE_LIB_OK=TRUE streamlit run app.py
> ```
> *(This is also handled within the project's runtime configuration).*

The application will launch in your browser at `http://localhost:8501`.

---

## 🔍 Interactive UI & Retrieval Debug Mode

The Streamlit UI is built not just as a chat client, but as an **interactive educational and diagnostic workbench**:

1. **Document Management Center**:
   - Live status indicator displaying current child and parent chunk counts.
   - Built-in drag-and-drop file uploader supporting `.pdf`, `.docx`, and `.txt`.
   - Action buttons for **Save Uploaded Files**, **Build Vector DB**, **Update Vector DB** (incremental delta addition skipping duplicates), and **Clear / Rebuild Index**.
2. **Dynamic Vector Engine Switching**:
   - Live toggle between **FAISS** (in-memory local storage) and **Pinecone** (distributed cloud index) directly in the sidebar without editing `.env`.
3. **White-Box "Retrieval Debug Mode"**:
   - Toggle **"Show Retrieval Details"** to inspect the pipeline's internal transformation stages for every response:
     - **Original Query**: The raw question entered by the user.
     - **Multi-Query Variations**: The generated semantic queries and coreference rewrites.
     - **Retrieved Candidates**: Chunks surfaced by the hybrid RRF search.
     - **Post-Reranker Scoring**: Candidates scored by the neural cross-encoder with relevance weights.
     - **Final Synthesized Context**: The exact token-budgeted prompt context dispatched to OpenAI.
4. **Strict Grounded Citations**:
   - Every generated answer prints linked, clickable citations specifying exact document names, sections, and page numbers.

---

## 🔮 Enterprise Scaling & Future Enhancements

To take this architecture from a production-grade prototype to a distributed, planet-scale enterprise system:

- [ ] **Quantitative RAG Evaluation (Ragas / TruLens)**: Integrate an automated evaluation harness measuring **Faithfulness**, **Answer Relevance**, **Context Precision**, and **Context Recall** against synthetic golden datasets.
- [ ] **Semantic Response Caching (Redis / GPTCache)**: Cache high-similarity semantic embeddings of user queries to serve instant cached responses for repetitive enterprise queries, saving up to 60% of API token expenditure.
- [ ] **Distributed Tracing & Observability**: Instrument the retrieval pipeline with **LangSmith** or **Arize Phoenix** for latency waterfall analysis, token tracking, and drift detection.
- [ ] **Asynchronous Chunk Streaming**: Leverage `StreamingResponse` via Server-Sent Events (SSE) to render tokens in real-time to the client interface.
- [ ] **Agentic Corrective Retrieval (CRAG)**: Implement an evaluative reflection loop that assesses retrieval relevance score confidence; if confidence falls below threshold, trigger an external fallback web search (e.g., Tavily API).

---

## 📄 License

This project is licensed under the terms of the [MIT License](LICENSE).

---

<p align="center">
  <b>Developed by Arpit Parmar</b><br>
  <i>Advanced Retrieval-Augmented Generation & Agentic AI Systems</i>
</p>
