# Agentic Secure Qdrant RAG API

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-backend-green)
![LangGraph](https://img.shields.io/badge/LangGraph-agentic-purple)
![Qdrant](https://img.shields.io/badge/Qdrant-vector--database-red)
![License](https://img.shields.io/badge/license-internal-lightgrey)

A **production-grade Retrieval-Augmented Generation (RAG) backend** built with **LangGraph**, **Qdrant**, and **FastAPI**.

This system is designed as a **secure headless microservice** that powers internal AI assistants. It features **agentic decision-making, hybrid retrieval, strict RBAC enforcement, and persistent memory**.

---

# Overview

This project implements a **secure agentic RAG architecture** where an LLM intelligently decides when to retrieve internal knowledge.

Key capabilities include:

- Agentic ReAct reasoning with LangGraph
- Hybrid dense + sparse retrieval
- Backend-enforced role-based access control
- Cross-encoder reranking
- Recursive session memory
- RAG evaluation and KPI tracking

The system is designed to be integrated with **web applications, internal dashboards, or enterprise assistants**.

---

# System Architecture

```
User Request
     │
     ▼
FastAPI API Layer
     │
     ▼
LangGraph Agent (ReAct Workflow)
     │
     ├── Decide: Answer directly
     │
     └── Decide: Call Retrieval Tool
                │
                ▼
        RBAC-secured search
                │
                ▼
     Hybrid Search (Dense + BM25)
                │
                ▼
        RRF Rank Fusion
                │
                ▼
      Cross Encoder Reranker
                │
                ▼
           LLM Response
                │
                ▼
        Session Memory Update
```

---

# Core Features

## Agentic ReAct Workflow

Uses **LangGraph ReAct agents** to dynamically decide whether to:

- Query the internal knowledge base
- Answer directly from conversation context

This significantly improves **efficiency and reasoning quality**.

---

## Security-First RBAC

Role-based access control is **strictly enforced on the backend**.

Supported roles:

- `admin`
- `employee`
- `volunteer`

Security parameters are injected into tool closures so the **LLM cannot manipulate access controls**.

---

## Hybrid Retrieval (Dense + Sparse)

The retrieval pipeline combines:

| Method           | Technology                |
| ---------------- | ------------------------- |
| Dense Retrieval  | Ollama `nomic-embed-text` |
| Sparse Retrieval | BM25 via FastEmbed        |

Results are merged using:

**Reciprocal Rank Fusion (RRF)**

This improves both **recall and ranking accuracy**.

---

## Cross-Encoder Reranking

A second ranking pass filters noisy results before context reaches the LLM.

Model used:

```
BAAI/bge-reranker-base
```

Benefits:

- Removes irrelevant chunks
- Improves factual grounding
- Reduces hallucinations

---

## Recursive Session Memory

A **dual-layer memory architecture**.

### Short-Term Memory

Stored in:

```
SQLite
```

Used for:

- session isolation
- exact fragment recall

---

### Long-Term Memory

Implemented using **LLM-powered recursive summarization**.

Stored as:

```
Markdown files
```

Advantages:

- Prevents context window overflow
- Preserves historical knowledge
- Enables long-running sessions

---

# Technology Stack

| Layer            | Technology                               |
| ---------------- | ---------------------------------------- |
| LLM              | OpenAI GPT (via NVIDIA NIM / OpenRouter) |
| Agent Framework  | LangChain + LangGraph                    |
| API              | FastAPI                                  |
| Server           | Uvicorn                                  |
| Vector Database  | Qdrant                                   |
| Embeddings       | Ollama (`nomic-embed-text`)              |
| Sparse Retrieval | FastEmbed BM25                           |
| Reranker         | Sentence Transformers Cross Encoder      |
| Session Storage  | SQLite                                   |

---

# Project Structure

```
project-root
│
├── improved_and_optimized_RAG.py
│   Main API server and agentic RAG pipeline
│
├── micro_rag_memory.py
│   Recursive memory system and SQLite session storage
│
├── evaluate_rag.py
│   Retrieval evaluation script
│
├── evaluate_tool_calling.py
│   Tests agent tool-calling accuracy
│
├── final_kpi_evaluation.py
│   Generates evaluation metrics
│
├── reindex_bm25.py
│   Initializes sparse vectors in Qdrant
│
├── DB_reset.py
│   Clears and reinitializes Qdrant collections
│
└── requirements.txt
```

# Project Structure & File Overview

The repository is modularized into core application logic, memory management, background processing, and a robust evaluation suite.

### 🧠 Core API & Agent Logic

- **`improved_and_optimized_RAG.py`**
  - **Role:** The main FastAPI server and orchestrator.
  - **Details:** Initializes the `QdrantClient`, `OllamaEmbeddings`, FastEmbed BM25 sparse model, and the BAAI Cross-Encoder. It defines the `search_internal_database` tool (with backend RBAC filters) and constructs the LangGraph ReAct agent.
  - **Endpoint:** Exposes the `POST /api/chat` route for frontend integrations.

- **`micro_rag_memory.py`**
  - **Role:** The dual-tier memory engine.
  - **Details:** Manages the SQLite vector database (`chat_memory.sqlite`) using `sqlite-vec` for short-term, exact-match conversational recall. It also contains the `summarize_and_rotate` function, which triggers a background LLM process to recursively condense SQL logs into permanent Markdown files (`session_summaries/`) to prevent context window bloat.

### ⚙️ Background Workers & Database Management

- **`reindex_bm25.py`**
  - **Role:** Asynchronous sparse vector generator.
  - **Details:** A lightweight Flask application running on port 5000. It safely scrolls through all existing dense vectors in Qdrant and generates/upserts the missing BM25 sparse vectors for Hybrid Search. This allows the primary embedding pipeline to remain fast.
  - **Endpoint:** Listens for a `POST /trigger-reindex` command to execute the job in a background thread.

- **`DB_reset.py`**
  - **Role:** Database initialization and hard-reset utility.
  - **Details:** Safely wipes the existing Qdrant collections and recreates them with exact configurations. It provisions `app_rag_docs` with hybrid parameters (Dense: 768 size, Sparse: IDF modifier) and initializes the `app_rag_file_registry` collection.

### 📊 Evaluation & KPI Suite

- **`compare_rankers.py`**
  - **Role:** Retrieval benchmark testing.
  - **Details:** Simulates and compares the legacy custom Python-side reranking architecture against the new Qdrant Native retrieval thresholding. Generates a table comparing latency (speed) and the number of returned documents per role.
- **`evaluate_tool_calling.py`**
  - **Role:** Agent behavior and security testing.
  - **Details:** Tests the LangGraph ReAct agent against trick questions, general knowledge prompts, and restricted queries. It verifies if the agent hallucinates tool calls, properly triggers the database search for factual questions, and correctly outputs security refusals when restricted.

- **`evaluate_rag.py`**
  - **Role:** Quality assurance (LLM-as-a-Judge).
  - **Details:** Uses a strictly-prompted LLM to read the final RAG outputs and grade them on **Faithfulness** (no hallucinations beyond the context) and **Relevance** (did it answer the user's question), outputting a pass/fail matrix.

- **`final_kpi_evaluation.py`**
  - **Role:** Executive metrics reporting.
  - **Details:** Runs a comprehensive test suite to calculate high-level performance indicators. Outputs the system's **Security Accuracy** (did RBAC thresholds hold), **Semantic Precision**, **Semantic Recall**, and the overall **F1 Quality Score**.

### 📦 Dependencies

- **`requirements.txt`**
  - **Role:** Python package definitions.
  - **Details:** Contains all necessary libraries including `langchain`, `langchain-openai`, `qdrant-client`, `fastapi`, `sqlite-vec`, `fastembed`, and `sentence-transformers`.

---

# Prerequisites

Before running the system ensure the following are installed.

## Python

```
Python 3.10+
```

---

## Qdrant Vector Database

Default local instance:

```
localhost:6333
```

Docker example:

```
docker run -p 6333:6333 qdrant/qdrant
```

---

## Ollama

Install **Ollama** and pull the required embedding model.

```
ollama pull nomic-embed-text:v1.5
```

⚠️ **Important**

The project explicitly uses the **`nomic-embed-text:v1.5`** model version.

If the version tag is omitted, Ollama defaults to the `:latest` tag.
This can cause **"Model Not Found" (404) errors** because the application code expects the exact model name **`nomic-embed-text:v1.5`**.

---

## API Key

Required for LLM access.

---

# Installation

## Clone Repository

```
git clone <repository-url>
cd KCS-CHATBOT-WITH-NEW-APPORACH
```

---

## Create Virtual Environment

### Windows

```
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```
python -m venv venv
source venv/bin/activate
```

---

## Install Dependencies

```
pip install -r requirements.txt
```

---

## Environment Configuration

Create a `.env` file:

```
NVIDIA_API_KEY=your_api_key_here
```

---

# Running the API

## Local Development

Start the API server:

```
python improved_and_optimized_RAG.py
```

Server will run at:

```
http://0.0.0.0:7860
```

---

## Docker Deployment

```
docker-compose up -d --build
```

---

# API Documentation

## Chat Endpoint

```
POST /api/chat
```

---

## Request Headers

```
Content-Type: application/json
```

---

## Example Request

```json
{
  "user_input": "What are the rules for volunteering?",
  "role": "volunteer",
  "session_id": "user-1234-session"
}
```

---

## Example Response

```json
{
  "reply": "Based on the internal documents, the rules for volunteering include completing the orientation program and adhering to the confidentiality agreement.",
  "debug_context": "--- [Relevance: 0.85] --- Volunteers must adhere to the confidentiality agreement...",
  "session_id": "user-1234-session"
}
```

If `session_id` is not provided, the system automatically generates one.

---

# Evaluation

The project includes evaluation pipelines for measuring RAG quality.

Metrics include:

- Precision
- Recall
- F1 Score
- Tool Invocation Accuracy
- Latency

Run evaluation:

```
python evaluate_rag.py
python final_kpi_evaluation.py
```

---

# Security Considerations

This system was designed with **security-first architecture**.

Key protections include:

- Backend-enforced RBAC
- Role injection via tool closures
- Controlled context injection
- Reranking safeguards against malicious context
- Session isolation with SQLite

---
