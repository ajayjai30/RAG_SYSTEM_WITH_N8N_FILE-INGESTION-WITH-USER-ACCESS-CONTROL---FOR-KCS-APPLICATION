# 🤖 Agentic Secure Qdrant RAG Chatbot

An advanced, production-grade Retrieval-Augmented Generation (RAG) system built with **LangGraph**, **Qdrant**, and **Gradio**. This chatbot features a strict agentic workflow, hybrid semantic/sparse search, backend-enforced Role-Based Access Control (RBAC), and session-isolated memory.

## 🌟 Key Features

-   **Agentic ReAct Workflow**: Uses LangGraph's ReAct agent to intelligently decide when to search the internal database versus answering based on conversation context.
-   **Security-First RBAC**: Role-Based Access Control is enforced at the backend level. The user's role (admin, employee, volunteer) is injected into the search tool's closure, ensuring the LLM never sees or manipulates security parameters.
-   **Hybrid Search with RRF**: Combines dense semantic embeddings (Ollama/Nomic) with sparse BM25 vectors using Reciprocal Rank Fusion (RRF) for superior retrieval precision.
-   **Cross-Encoder Reranking**: Leverages `BAAI/bge-reranker-base` to audit and rerank retrieved chunks, filtering out noise and ensuring only the most relevant context reaches the LLM.
-   **Micro-RAG Session Memory**: A session-isolated SQLite memory system with 48-hour TTL, providing persistent conversational context without context-window bloat.
-   **Performance Monitoring**: Integrated evaluation scripts for tracking RAG quality, tool-calling accuracy, and overall KPI metrics.

## 🛠️ Tech Stack

-   **LLM**: OpenAI GPT models (via NVIDIA NIM/OpenRouter)
-   **Orchestration**: LangChain, LangGraph
-   **Vector DB**: Qdrant (Hybrid Search: Dense + Sparse)
-   **Embeddings**: Ollama (nomic-embed-text), FastEmbed (BM25)
-   **Reranker**: Sentence-Transformers (Cross-Encoder)
-   **UI**: Gradio 5+
-   **Database**: SQLite (Session Memory)

## 🚀 Getting Started

### Prerequisites

-   Python 3.10+
-   Qdrant Server running (default: `localhost:6333`)
-   Ollama running with `nomic-embed-text`
-   NVIDIA API Key (for LLM access)

### Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd KCS-CHATBOT-WITH-NEW-APPORACH
    ```

2.  **Set up Virtual Environment**:
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configuration**:
    Create a `.env` file or update the configuration block in `improved_and_optimized_RAG.py` with your API keys and server URLs.

### Running the App

```bash
python improved_and_optimized_RAG.py
```

## 📂 Project Structure

-   `improved_and_optimized_RAG.py`: The main application script featuring the agentic RAG pipeline and Gradio UI.
-   `micro_rag_memory.py`: Implementation of the SQLite-based session isolation and memory logic.
-   `evaluate_rag.py` / `evaluate_tool_calling.py`: Automated testing suites for measuring retrieval and agentic performance.
-   `final_kpi_evaluation.py`: Generates comprehensive metric reports (F1, Precision, Latency).
-   `reindex_bm25.py`: Utility script for initializing sparse vectors in Qdrant.

## 📊 Evaluation & Verification

To run the full evaluation suite and generate KPI reports:

```bash
python evaluate_rag.py
python final_kpi_evaluation.py
```

---
*Built for secure, internal company assistance.*
