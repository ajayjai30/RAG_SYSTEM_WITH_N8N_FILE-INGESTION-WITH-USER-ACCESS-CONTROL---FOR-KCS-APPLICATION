import uuid
import threading
import os
import micro_rag_memory

# FastAPI Imports
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Global counter for rotation trigger
query_counter = 0

# Qdrant & Embedding Imports
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_community.embeddings import OllamaEmbeddings
from fastembed import SparseTextEmbedding
from sentence_transformers import CrossEncoder

# LangChain / LangGraph Imports
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

# ==========================================
#  CONFIGURATION
# ==========================================

API_KEY = "insert_api_key_for_llm_here" # Replace with API key
LLM_MODEL_NAME = "openai/gpt-oss-20b" #or any model u wish
BASE_URL="the_base_url_of_the_cloud_provider_for_LLM"
LOCAL_EMBED_URL = "http://ollama:11434" # Docker network URL
LOCAL_EMBED_MODEL = "nomic-embed-text:v1.5"

RERANK_THRESHOLD = 0.55
QDRANT_COLLECTION = "app_rag_docs"

client = QdrantClient("http://qdrant:6333") # Docker network URL
dense_embeddings = OllamaEmbeddings(base_url=LOCAL_EMBED_URL, model=LOCAL_EMBED_MODEL)
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
reranker = CrossEncoder('BAAI/bge-reranker-base')

llm = ChatOpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    model=LLM_MODEL_NAME
)

# ==========================================
#  TOOLS
# ==========================================

@tool
def search_internal_database(query: str, role: str):
    """Search the company internal database. Role-Based Access Control (RBAC) is applied automatically."""
    search_prompt = f"search_query: {query}"
    dense_vector = dense_embeddings.embed_query(search_prompt)
    sparse_vector = list(sparse_model.embed([query]))[0]
    
    role_filter = models.Filter(
        must=[models.FieldCondition(key="access_role", match=models.MatchValue(value=role))]
    )
    
    search_results = client.query_points(
        collection_name=QDRANT_COLLECTION,
        prefetch=[
            models.Prefetch(query=dense_vector, using="", limit=15),
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse_vector.indices.tolist(),
                    values=sparse_vector.values.tolist(),
                ),
                using="sparse", limit=15,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        query_filter=role_filter,
        limit=10,
    )
    
    if not search_results.points:
        return "No relevant documents found in the database."

    docs = [p.payload.get("content", "") for p in search_results.points]
    pairs = [[query, doc] for doc in docs]
    scores = reranker.predict(pairs)
    
    ranked_results = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    
    final_context = ""
    for doc, score in ranked_results:
        if score > RERANK_THRESHOLD:
            final_context += f"--- [Relevance: {score:.2f}] ---\n{doc}\n\n"
            
    return final_context if final_context else "Information found, but failed relevance threshold."

# ==========================================
#  AGENT LOGIC
# ==========================================

def process_query(user_input: str, role: str, session_id: str):
    global query_counter
    query_counter += 1
    
    # 1. RETRIEVE MEMORIES (Markdown + SQL)
    long_term_summary = micro_rag_memory.get_existing_summary(session_id)
    short_term_context = micro_rag_memory.get_memories(session_id, user_input)

    # 2. CONSTRUCT SYSTEM PROMPT
    system_instruction = f"""You are a secure corporate AI assistant.
    User Security Role: {role}

    PERSISTENT SUMMARY OF PREVIOUS TURNS:
    {long_term_summary}

    RECENT RELEVANT FRAGMENTS:
    {short_term_context}

    INSTRUCTIONS:
    - Use the summaries above to maintain context across days/sessions.
    - If a user refers to 'the document' or 'the project', check the summary first.
    - Always strictly follow RBAC filters for the {role} role.
    """

    agent = create_react_agent(
        llm, 
        tools=[search_internal_database], 
        state_modifier=SystemMessage(content=system_instruction)
    )
    
    response = agent.invoke({"messages": [HumanMessage(content=user_input)]})
    bot_message = response["messages"][-1].content

    raw_tool_context = "No database search was triggered."
    for msg in response["messages"]:
        if msg.type == "tool":
            raw_tool_context = msg.content

    # 3. SAVE TO SHORT-TERM MEMORY (SQL)
    micro_rag_memory.add_memory(session_id, user_input, bot_message)

    # 4. TRIGGER ROTATION (Every 5 queries)
    if query_counter % 5 == 0:
        threading.Thread(
            target=micro_rag_memory.summarize_and_rotate, 
            args=(session_id, llm)
        ).start()

    return {
        "reply": bot_message,
        "debug_context": raw_tool_context,
        "session_id": session_id
    }

# ==========================================
#  FASTAPI APP DEFINITION
# ==========================================

app = FastAPI(title="KCS Agentic RAG API", version="1.0")

class ChatRequest(BaseModel):
    user_input: str
    role: str
    session_id: str = None # Optional: If none is provided, generate a new one

@app.post("/api/chat")
def chat_endpoint(request: ChatRequest):
    try:
        # Generate a new session ID if the frontend didn't provide one
        session_id = request.session_id if request.session_id else str(uuid.uuid4())
        
        result = process_query(
            user_input=request.user_input, 
            role=request.role, 
            session_id=session_id
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("Initializing Database...")
    micro_rag_memory.init_db()
    print("Starting FastAPI Server...")
    # Runs the server on port 7860 so you don't have to change your docker-compose.yml
    uvicorn.run(app, host="0.0.0.0", port=7860)