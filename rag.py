import gradio as gr
import json
import os
import uuid
import micro_rag_memory

# Qdrant Imports
from qdrant_client import QdrantClient
from qdrant_client.http import models

# LangChain Imports
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder
from fastembed import SparseTextEmbedding

# Add BGE Reranker
reranker = CrossEncoder('BAAI/bge-reranker-base')
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

# ==========================================
#  CONFIGURATION 
# ==========================================

# LLM Config
OPENROUTER_API_KEY = "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" 
LLM_MODEL_NAME     = "liquid/lfm-2.5-1.2b-thinking:free"

# Embedding Config
LOCAL_EMBED_URL   = "http://localhost:11434"
LOCAL_EMBED_MODEL = "nomic-embed-text:v1.5"

# ==========================================
#  AI & DB INITIALIZATION
# ==========================================

# Connect to local Qdrant container
qdrant = QdrantClient("http://localhost:6333")

# Note: We are using the community embeddings here. 
# If you get a deprecation warning, it is harmless and will still run perfectly!
embeddings = OllamaEmbeddings(
    base_url=LOCAL_EMBED_URL, 
    model=LOCAL_EMBED_MODEL
)

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    model=LLM_MODEL_NAME,
    temperature=0.3
)

template = """
You are a secure internal assistant. Answer the user's question based strictly and ONLY on the "RETRIEVED DATABASE CONTEXT" below. 
Do not use any outside knowledge to answer the question. If the answer cannot be explicitly found in the context, or if the context is empty, you MUST exactly say: "I don't have access to that information based on your current security clearance."
Do not attempt to guess, extrapolate, or hallucinate an answer.

### CONVERSATIONAL MEMORY AND RECENT CHAT
{memories}
{recent_messages}

### RETRIEVED DATABASE CONTEXT
{context}

### QUESTION: 
{question}
"""
prompt = ChatPromptTemplate.from_template(template)
rag_chain = prompt | llm | StrOutputParser()

guardrail_template = """
You are a security guardrail. Your job is to check if the user is authorized to ask their question based on their role.
Roles: 
- admin: Authorized for EVERYTHING.
- employee: Authorized to ask about employee and volunteer duties, operations, schedules.
- volunteer: Authorized to ask ONLY about volunteer duties, public info, rules. NOT authorized to ask about employee wages, employee schedules, or sensitive admin data.

User Role: {role}
Question: {question}

Reply exactly with "ALLOW" if they are authorized, or "DENY" if they are asking about topics restricted from their role. Do not explain.
"""
guardrail_chain = ChatPromptTemplate.from_template(guardrail_template) | llm | StrOutputParser()

# ==========================================
#  NATIVE QDRANT RETRIEVAL LOGIC
# ==========================================

def search_qdrant(query_text, user_role):
    """
    Retrieves data using Qdrant Payload Filtering + Instruction-Aware Embeddings.
    No external reranker needed. Replaces Oracle VPD.
    """
    try:
        # 1. Prepend the search query instruction for high-accuracy asymmetric retrieval
        optimized_query = f"search_query: {query_text}"
        query_vector = embeddings.embed_query(optimized_query)

        # 1b. Compute Sparse Vector for BM25 mapping
        try:
            sparse_emb = list(sparse_model.embed([query_text]))[0]
            sparse_vector_query = models.SparseVector(
                indices=sparse_emb.indices.tolist(), 
                values=sparse_emb.values.tolist()
            )
        except Exception as e:
            print("BM25 encoding failed", e)
            sparse_vector_query = None

        # 2. Security Handshake (Role-Based Payload Filtering)
        role_filter = None
        if user_role == 'employee':
            # Employee sees Employee AND Volunteer data
            role_filter = models.Filter(
                must=[models.FieldCondition(key="access_role", match=models.MatchAny(any=["employee", "volunteer"]))]
            )
        elif user_role == 'volunteer':
            # Volunteer sees ONLY Volunteer data
            role_filter = models.Filter(
                must=[models.FieldCondition(key="access_role", match=models.MatchValue(value="volunteer"))]
            )
        elif user_role == 'admin':
            # Admin sees everything (No filter)
            role_filter = None
        else:
            # DEFAULT DENY
            return ""

        # 3. Execute Native High-Accuracy Hybrid Search
        if sparse_vector_query:
            prefetch = [
                models.Prefetch(
                    query=query_vector,
                    filter=role_filter,
                    limit=20,
                ),
                models.Prefetch(
                    query=sparse_vector_query,
                    using="sparse",
                    filter=role_filter,
                    limit=20,
                )
            ]
            results = qdrant.query_points(
                collection_name="app_rag_docs",
                prefetch=prefetch,
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=10,
                with_payload=True
            ).points
        else:
            results = qdrant.query_points(
                collection_name="app_rag_docs",
                query=query_vector,
                query_filter=role_filter,
                limit=10, 
                with_payload=True
            ).points

        if not results:
            return ""

        # Cross-Encoder Reranking
        doc_pairs = [[query_text, hit.payload.get('content', '')] for hit in results]
        scores = reranker.predict(doc_pairs)
        
        # Zip hits with scores and sort
        scored_results = list(zip(results, scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        # Take Top 4 Reranked docs
        top_results = scored_results[:4]

        # 4. Format Output
        context_parts = []
        for hit, score in top_results:
            # We skip the raw score threshold since the Reranker is highly accurate
            score_display = round(score, 3) 
            context_parts.append(f"[Reranker Logit: {score_display} | Security: {hit.payload.get('access_role', 'unknown')}]\n{hit.payload.get('content', '')}")
            
        return "\n\n---\n\n".join(context_parts)

    except Exception as e:
        return f"Database Error: {str(e)}"

# ==========================================
# CORE LOGIC FOR QUERY PROCESSING
# ==========================================

def process_query(message, history, user_role, session_id):
    # Initialize history if None
    if history is None:
        history = []
    
    if not message:
        return history, "", "", session_id

    # 1. Pre-Retrieval LLM Guardrail (Intent & Role Routing)
    try:
        auth_decision = guardrail_chain.invoke({"role": user_role, "question": message}).strip()
        if "DENY" in auth_decision.upper():
            bot_response = f" 🔒 ACCESS DENIED: Your role ('{user_role}') is not authorized to query about this topic."
            debug_output = "System Log: Pre-Retrieval LLM Guardrail blocked the query (Intent matching failed authorization)."
            new_history = history + [[message, bot_response]]
            return new_history, debug_output, "", session_id
    except Exception as e:
        print(f"Guardrail error: {e}") # Fail open if LLM fails just in case

    # 2. Retrieve Context from Qdrant
    retrieved_docs = search_qdrant(message, user_role)
    
    # 2. Get Past Memories
    past_memories_str = micro_rag_memory.get_memories(session_id, message)
    
    # 3. Format Last 2 Messages for context (Classic Tuple format to prevent Gradio errors)
    recent_messages_str = ""
    if history:
        recent_messages_str += "--- LAST 2 MESSAGES ---\n"
        for turn in history[-2:]:
            recent_messages_str += f"User: {turn[0]}\nAssistant: {turn[1]}\n"
    
    # 4. Handle "No Access" Case
    if not retrieved_docs:
        bot_response = f" 🔒 ACCESS DENIED: The database blocked all documents for role '{user_role}'."
        debug_output = "System Log: Qdrant Payload Filter returned 0 rows. \n(User has no permission to see relevant data)."
        
        new_history = history + [[message, bot_response]]
        return new_history, debug_output, "", session_id

    # 5. Generate Answer
    try:
        bot_response = rag_chain.invoke({
            "context": retrieved_docs,
            "memories": past_memories_str,
            "recent_messages": recent_messages_str,
            "question": message
        })
        # Save memory AFTER generation
        micro_rag_memory.add_memory(session_id, message, bot_response)
        
    except Exception as e:
        bot_response = f"LLM Error: {str(e)}"

    # 6. Update UI (Classic Tuple format)
    new_history = history + [[message, bot_response]]
    return new_history, retrieved_docs, "", session_id 

# ==========================================
# GRADIO UI 
# ==========================================

custom_css = """
footer {visibility: hidden}
.debug-box textarea {font-family: monospace; font-size: 12px; background-color: #f0f0f0;}
"""

with gr.Blocks(title="Secure Qdrant RAG") as demo:
    session_id = gr.State(lambda: str(uuid.uuid4()))
    
    gr.Markdown("# Secure RAG: User Restricted Database Access with Qdrant")
    gr.Markdown("Inspect the 'Invisible Wall' in real-time. See exactly what the LLM sees.")

    with gr.Row():
        # --- LEFT COLUMN: Chat Interface ---
        with gr.Column(scale=5):
            # Removed type="messages" to support older versions of Gradio perfectly
            chatbot = gr.Chatbot(
                label="Assistant", 
                height=500
            )
            
            with gr.Row():
                role_selector = gr.Dropdown(
                    choices=["admin", "employee", "volunteer"], 
                    value="volunteer", 
                    label=" Current Identity",
                    info="Simulates login. Updates DB Context.",
                    interactive=True
                )
            
            msg = gr.Textbox(
                label="Your Question", 
                placeholder="Ask about protocols, schedules, or sensitive data...",
                lines=2
            )
            
            submit_btn = gr.Button("Ask Database", variant="primary")
            clear_btn = gr.Button("Clear History")

        # --- RIGHT COLUMN: Debug/Context View ---
        with gr.Column(scale=4):
            gr.Markdown("### 🔍 Retrieval Debugger")
            gr.Markdown("*This shows the Raw Text returned by Qdrant BEFORE the LLM sees it.*")
            
            debug_panel = gr.TextArea(
                label="Retrieved Context (Raw Database Output)", 
                placeholder="Context will appear here after query...",
                lines=25,
                interactive=False, 
                elem_classes="debug-box"
            )

    # --- WIRING THE EVENTS ---
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

    clear_btn.click(lambda: ([], "", "", str(uuid.uuid4())), None, [chatbot, debug_panel, msg, session_id])

if __name__ == "__main__":
    print("Starting Secure Qdrant RAG Debugger...")
    demo.launch(share=False, css=custom_css, theme=gr.themes.Soft(), debug=True)