import gradio as gr
import uuid
import micro_rag_memory
import threading

# Global counter for TTL cleanup trigger
query_counter = 0

# Qdrant Imports
from qdrant_client import QdrantClient
from qdrant_client.http import models

# LangChain / LangGraph Imports
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

# Cross-Encoder Reranker
from sentence_transformers import CrossEncoder

# BM25 Sparse Embeddings
from fastembed import SparseTextEmbedding

# ==========================================
#  CONFIGURATION
# ==========================================

# LLM Config
NVIDIA_API_KEY = "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"
LLM_MODEL_NAME     = "openai/gpt-oss-20b"

# Embedding Config
LOCAL_EMBED_URL   = "http://localhost:11434"
LOCAL_EMBED_MODEL = "nomic-embed-text:v1.5"

# --- Quality Thresholds ---
# FIX: Increased from 0.45 → 0.55 to cut retrieval noise (50% Semantic Precision issue)
NATIVE_SCORE_THRESHOLD = 0.55

# Cross-Encoder: drop chunks whose logit is at or below this value (0.0 = decision boundary)
RERANKER_LOGIT_THRESHOLD = 0.0

# ==========================================
#  AI & DB INITIALIZATION
# ==========================================

qdrant = QdrantClient("http://localhost:6333")

embeddings = OllamaEmbeddings(
    base_url=LOCAL_EMBED_URL,
    model=LOCAL_EMBED_MODEL
)

llm = ChatOpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY,
    model=LLM_MODEL_NAME,
    temperature=0.3
)

# Cross-Encoder Reranker — BAAI/bge-reranker-base
reranker = CrossEncoder('BAAI/bge-reranker-base')

# BM25 Sparse model
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

# ==========================================
#  STRICT COMPANY-ONLY SYSTEM PROMPT
# ==========================================

STRICT_SYSTEM_PROMPT = """You are a highly secure, strict internal assistant for the company. 
You have access to a tool called `search_internal_database`. 

YOUR STRICT BEHAVIOR RULES:
1. EXCLUSIVE SCOPE: You are exclusively an internal company assistant. You MUST NOT answer general knowledge questions, write code, or discuss topics outside of the company. 
2. OFF-TOPIC HANDLING: If a user asks a question unrelated to the company, you must refuse to answer and reply EXACTLY with: "I am a strict internal company assistant. I can only answer questions related to our internal data."
3. INTERNAL KNOWLEDGE: For ANY question asking for facts, policies, schedules, employee data, or guidelines, you MUST use the `search_internal_database` tool. 
4. HANDLING DENIALS & MISSING DATA: The database is strictly role-restricted. If the tool returns "ACCESS DENIED" or if the retrieved context is completely irrelevant to the question, you MUST NOT guess. You must reply EXACTLY with: "I don't have access to that information based on your current security clearance or the data is missing." """

# ==========================================
#  NATIVE QDRANT RETRIEVAL LOGIC
# ==========================================

def search_qdrant(query_text: str, user_role: str, return_raw: bool = False):
    """
    Retrieves data using Qdrant Payload Filtering + Hybrid Search + Cross-Encoder Reranking.

    Security architecture:
    - `user_role` is NEVER exposed to the LLM. Injected by the backend closure only.
    - Role filtering is enforced at the Qdrant payload level.
    - Cross-encoder reranker drops low-quality chunks (logit <= RERANKER_LOGIT_THRESHOLD).

    return_raw=True → returns raw dicts for KPI evaluation (final_kpi_evaluation.py).
    """
    try:
        # 1. Instruction-aware embedding for asymmetric retrieval
        optimized_query = f"search_query: {query_text}"
        query_vector = embeddings.embed_query(optimized_query)

        # 1b. Sparse BM25 vector for hybrid search
        sparse_vector_query = None
        try:
            sparse_emb = list(sparse_model.embed([query_text]))[0]
            sparse_vector_query = models.SparseVector(
                indices=sparse_emb.indices.tolist(),
                values=sparse_emb.values.tolist()
            )
        except Exception as e:
            print(f"[BM25] Sparse encoding failed: {e}")

        # 2. Backend Security Handshake — Role-Based Payload Filtering
        #    The LLM tool only passes `query`; role is injected here in the backend.
        if user_role == 'employee':
            role_filter = models.Filter(
                must=[models.FieldCondition(
                    key="access_role",
                    match=models.MatchAny(any=["employee", "volunteer"])
                )]
            )
        elif user_role == 'volunteer':
            role_filter = models.Filter(
                must=[models.FieldCondition(
                    key="access_role",
                    match=models.MatchValue(value="volunteer")
                )]
            )
        elif user_role == 'admin':
            role_filter = None  # Admins see everything
        else:
            # DEFAULT DENY — unknown role
            return [] if return_raw else "ACCESS DENIED: Unknown role."

        # 3. Execute Hybrid Search (Dense + BM25 via RRF fusion) or Dense fallback
        if sparse_vector_query:
            prefetch = [
                models.Prefetch(query=query_vector, filter=role_filter, limit=20, score_threshold=NATIVE_SCORE_THRESHOLD),
                models.Prefetch(query=sparse_vector_query, using="sparse", filter=role_filter, limit=20),
            ]
            results = qdrant.query_points(
                collection_name="app_rag_docs",
                prefetch=prefetch,
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=10,
                with_payload=True
            ).points
        else:
            # Dense-only with native score threshold (0.55 — tightened)
            results = qdrant.query_points(
                collection_name="app_rag_docs",
                query=query_vector,
                query_filter=role_filter,
                limit=10,
                score_threshold=NATIVE_SCORE_THRESHOLD,
                with_payload=True
            ).points

        if not results:
            return [] if return_raw else ""

        # 4. Cross-Encoder Reranking — audit & filter BEFORE returning to LLM
        doc_pairs = [[query_text, hit.payload.get('content', '')] for hit in results]
        rerank_scores = reranker.predict(doc_pairs)

        # Sort high → low and drop chunks below the logit decision boundary
        scored = sorted(zip(results, rerank_scores), key=lambda x: x[1], reverse=True)
        filtered = [(hit, score) for hit, score in scored if score > RERANKER_LOGIT_THRESHOLD]
        top_results = filtered[:4]

        if not top_results:
            return [] if return_raw else ""

        # 5. Format output
        if return_raw:
            return [
                {
                    "content": hit.payload.get('content', ''),
                    "role": hit.payload.get('access_role', 'unknown'),
                    "score": round(float(score), 3)
                }
                for hit, score in top_results
            ]

        context_parts = []
        for hit, score in top_results:
            context_parts.append(
                f"[Reranker Logit: {round(score, 3)} | Security: {hit.payload.get('access_role', 'unknown')}]\n"
                f"{hit.payload.get('content', '')}"
            )
        return "\n\n---\n\n".join(context_parts)

    except Exception as e:
        print(f"[DB Error] {e}")
        return [] if return_raw else f"Database Error: {str(e)}"


# ==========================================
#  SECURE TOOL FACTORY (Closure-based Role Injection)
# ==========================================

def make_search_tool(current_user_role: str):
    """
    Returns a LangChain tool whose ONLY LLM-visible parameter is `query: str`.
    The `current_user_role` is captured in the closure — the LLM NEVER sees it.
    This enforces backend-only RBAC with zero role leakage to the model.
    """
    @tool
    def search_internal_database(query: str) -> str:
        """
        Search the internal company database for relevant information.
        Use this tool for ANY question about company facts, policies,
        schedules, employee data, volunteer duties, or internal guidelines.
        """
        result = search_qdrant(query, current_user_role)
        if not result:
            return "ACCESS DENIED: No documents found for your query under your current security clearance."
        return result

    return search_internal_database


# ==========================================
#  CORE LOGIC FOR QUERY PROCESSING
# ==========================================

async def process_query(message, history, user_role, session_id):
    """
    Agentic RAG pipeline using LangGraph's create_react_agent.
    - The LLM decides WHEN to call search_internal_database.
    - user_role is NEVER in the LLM's context — injected via closure only.
    - Returns history as [[user, bot], ...] tuples for evaluate_rag.py compatibility.
    """
    global query_counter
    query_counter += 1
    if query_counter % 10 == 0:
        threading.Thread(target=micro_rag_memory.cleanup_old_memories, daemon=True).start()

    if history is None:
        history = []
    if not message:
        return history, "", "", session_id

    # 1. Build role-injected search tool for this session
    search_tool = make_search_tool(user_role)

    # 2. Build LangGraph ReAct agent with strict system prompt
    agent = create_react_agent(
        model=llm,
        tools=[search_tool],
        prompt=STRICT_SYSTEM_PROMPT,
    )

    # 3. Build conversational context for the message
    past_memories_str = micro_rag_memory.get_memories(session_id, message)

    recent_messages_str = ""
    if history:
        recent_messages_str = "\n--- RECENT HISTORY ---\n"
        for turn in history[-4:]:
            if isinstance(turn, (list, tuple)) and len(turn) == 2:
                recent_messages_str += f"User: {turn[0]}\nAssistant: {turn[1]}\n"
            elif isinstance(turn, dict):
                role_label = turn.get("role", "unknown").capitalize()
                recent_messages_str += f"{role_label}: {turn.get('content', '')}\n"

    # Prepend memory context to the user message
    full_input = message
    if past_memories_str or recent_messages_str:
        full_input = f"{recent_messages_str}\n{past_memories_str}\n\nCurrent Question: {message}"

    # 4. Invoke agent
    retrieved_docs = ""
    try:
        result = await agent.ainvoke({"messages": [HumanMessage(content=full_input)]})

        # Extract final answer from last AI message
        all_messages = result.get("messages", [])
        bot_response = ""
        for msg in reversed(all_messages):
            # LangGraph returns AIMessage objects; get the last non-empty one
            has_tools = getattr(msg, "tool_calls", None)
            if hasattr(msg, "content") and msg.content and not has_tools:
                bot_response = msg.content.strip()
                break

        if not bot_response:
            bot_response = "I was unable to generate a response."

        # Extract tool observation (what the DB returned) for the debug panel
        for msg in all_messages:
            # ToolMessage objects carry the tool's return value
            if msg.__class__.__name__ == "ToolMessage" and msg.content:
                if not msg.content.startswith("ACCESS DENIED"):
                    retrieved_docs = msg.content

        micro_rag_memory.add_memory(session_id, message, bot_response)

    except Exception as e:
        bot_response = f"Agent Error: {str(e)}"
        retrieved_docs = ""

    # 5. Return dictionary-format history (required by Gradio 5+)
    new_history = list(history)
    new_history.extend([
        {"role": "user", "content": message}, 
        {"role": "assistant", "content": bot_response}
    ])
    return new_history, retrieved_docs, "", session_id


# ==========================================
#  GRADIO UI
# ==========================================

custom_css = """
footer {visibility: hidden}
.debug-box textarea {font-family: monospace; font-size: 12px; background-color: #f0f0f0;}
"""

with gr.Blocks(title="Agentic Secure Qdrant RAG") as demo:
    gr.Markdown("# 🤖 Agentic Secure RAG: Tool-Calling with Backend Role Security")
    gr.Markdown(
        "The LLM **decides when to search** via tool calling. "
        "The `user_role` is **never visible to the LLM** — it is injected in the backend tool closure."
    )

    with gr.Row():
        with gr.Column(scale=5):
            chatbot = gr.Chatbot(label="Assistant", height=500)

            with gr.Row():
                role_selector = gr.Dropdown(
                    choices=["admin", "employee", "volunteer"],
                    value="volunteer",
                    label=" Current Identity",
                    info="Simulates login. Role is injected backend-only — never seen by the LLM.",
                    interactive=True
                )
                session_id = gr.Textbox(
                    label="User / Session ID",
                    value="default-user-123",
                    info="Persists 48-hour history across page refreshes.",
                    interactive=True
                )

            msg = gr.Textbox(
                label="Your Question",
                placeholder="Ask about protocols, schedules, or sensitive data...",
                lines=2
            )
            submit_btn = gr.Button("Ask Database", variant="primary")
            clear_btn = gr.Button("Clear History")

        with gr.Column(scale=4):
            gr.Markdown("### 🔍 Retrieval Debugger")
            gr.Markdown("*Shows the raw context the tool returned BEFORE the LLM's final answer.*")
            debug_panel = gr.TextArea(
                label="Tool Retrieved Context (Raw Database Output)",
                placeholder="Context will appear here after the agent calls the tool...",
                lines=25,
                interactive=False,
                elem_classes="debug-box"
            )

    msg.submit(
        fn=process_query,
        inputs=[msg, chatbot, role_selector, session_id],
        outputs=[chatbot, debug_panel, msg, session_id]
    )
    submit_btn.click(
        fn=process_query,
        inputs=[msg, chatbot, role_selector, session_id],
        outputs=[chatbot, debug_panel, msg, session_id]
    )
    clear_btn.click(
        lambda sid: ([], "", "", sid),
        inputs=[session_id],
        outputs=[chatbot, debug_panel, msg, session_id]
    )

if __name__ == "__main__":
    print("Starting Agentic Secure Qdrant RAG...")
    demo.launch(share=False, css=custom_css, theme=gr.themes.Soft(), debug=True)
