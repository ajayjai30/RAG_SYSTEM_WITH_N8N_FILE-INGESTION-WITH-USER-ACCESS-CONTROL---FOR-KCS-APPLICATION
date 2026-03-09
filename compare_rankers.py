import json
import time
import os
import uuid
import math
from typing import List, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_community.embeddings import OllamaEmbeddings

# Local imports from current workspace
import rag

# ==========================================
#  CONFIGURATION & SETUP
# ==========================================

qdrant = QdrantClient("http://localhost:6333")
LOCAL_EMBED_URL   = "http://localhost:11434"
LOCAL_EMBED_MODEL = "nomic-embed-text:v1.5"

# We use the community embeddings here for exact equivalence to old system
embeddings = OllamaEmbeddings(
    base_url=LOCAL_EMBED_URL, 
    model=LOCAL_EMBED_MODEL
)

# Load Legacy EVAL results to extract test queries
try:
    with open(r"D:\KCS CODE FILES - Copy\rag_eval_results.json", "r") as f:
         LEGACY_RESULTS = json.load(f)
except Exception as e:
    print(f"FAILED to load legacy results: {e}")
    LEGACY_RESULTS = []

# ==========================================
#  PATH A: LEGACY CUSTOM RERANKER (Simulated)
# ==========================================
# Re-implementing D:\KCS CODE FILES - Copy\reranker_service.py logic

import requests

def get_embed(text: str, kind: str) -> list:
    prefixed = f"{kind}: {text}"
    payload = {"model": LOCAL_EMBED_MODEL, "input": prefixed}
    r = requests.post(f"{LOCAL_EMBED_URL}/api/embed", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()["embeddings"][0]

def cosine_similarity(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

def legacy_rerank(query: str, retrieved_docs: list) -> list:
    if not retrieved_docs: return []
    start_time = time.time()
    query_vec = get_embed(query, kind="search_query")
    
    scored = []
    for doc in retrieved_docs:
        try:
             doc_vec = get_embed(doc[:600], kind="search_document")
             score = cosine_similarity(query_vec, doc_vec)
             scored.append({"content": doc, "score": max(0.0, score)})
        except:
             scored.append({"content": doc, "score": 0.0})

    filtered = [d for d in scored if d["score"] >= 0.45] # Legacy threshold
    filtered.sort(key=lambda x: x["score"], reverse=True)
    return filtered[:4], time.time() - start_time


def run_legacy_pipeline(query: str, role: str) -> dict:
    """Simulates Legacy Oracle Retrieval + Reranker using Qdrant as the backend."""
    t0 = time.time()
    
    # 1. Oracle Wide Recall Simulation (Fetch Top 50, No strict threshold)
    query_vector = embeddings.embed_query(query)
    
    role_filter = None
    if role == 'employee':
        role_filter = models.Filter(must=[models.FieldCondition(key="access_role", match=models.MatchAny(any=["employee", "volunteer"]))])
    elif role == 'volunteer':
        role_filter = models.Filter(must=[models.FieldCondition(key="access_role", match=models.MatchValue(value="volunteer"))])
        
    # Use rag.qdrant to piggyback off its connection, or fix our local qdrant
    results = rag.qdrant.query_points(
        collection_name="app_rag_docs",
        query=query_vector,
        query_filter=role_filter,
        limit=50,  # Legacy fetched 50
        with_payload=True
    ).points
    
    if not results:
        return {"docs": 0, "latency": time.time() - t0, "top_score": 0}
        
    # Python distance filtering (legacy filtered distance < 0.6 => score > 0.4 approx)
    valid_docs = [hit.payload.get('content', '') for hit in results if hit.score > 0.4]
    
    rerank_time = 0
    final_docs = []
    top_score = 0
    
    # 2. Reranking (Python side)
    if valid_docs:
       final_results, rerank_time = legacy_rerank(query, valid_docs)
       final_docs = [r['content'] for r in final_results]
       top_score = final_results[0]['score'] if final_results else 0

    total_latency = time.time() - t0
    
    return {
        "docs": len(final_docs), 
        "latency": total_latency, 
        "rerank_time": rerank_time,
        "top_score": top_score
    }


# ==========================================
#  PATH B: QDRANT NATIVE (New System)
# ==========================================
def run_native_pipeline(query: str, role: str) -> dict:
    t0 = time.time()
    query_vector = embeddings.embed_query(f"search_query: {query}")
    
    role_filter = None
    if role == 'employee':
        role_filter = models.Filter(must=[models.FieldCondition(key="access_role", match=models.MatchAny(any=["employee", "volunteer"]))])
    elif role == 'volunteer':
        role_filter = models.Filter(must=[models.FieldCondition(key="access_role", match=models.MatchValue(value="volunteer"))])
        
    results = rag.qdrant.query_points(
        collection_name="app_rag_docs",
        query=query_vector,
        query_filter=role_filter,
        limit=4, 
        score_threshold=0.8, # Native Threshold
        with_payload=True
    ).points
    
    total_latency = time.time() - t0
    
    top_score = results[0].score if results else 0
    
    return {
        "docs": len(results), 
        "latency": total_latency, 
        "top_score": top_score
    }

# ==========================================
#  EXECUTION ENGINE
# ==========================================

def compare():
    print("="*80)
    print(" STARTING RAG RETRIEVAL COMPARISON: CUSTOM RERANKER VS NATIVE QDRANT")
    print("="*80)
    
    results_table = []
    
    queries_to_test = [
        {"role": "admin", "q": "What are the master passwords for the database?"},
        {"role": "volunteer", "q": "What are the master passwords for the database?"},
        {"role": "volunteer", "q": "What are the rules for volunteering?"},
        {"role": "employee", "q": "How do I wipe up data?"},
        {"role": "admin", "q": "How do I wipe up data?"}
    ]
    
    for idx, tc in enumerate(queries_to_test):
        q = tc['q']
        role = tc['role']
        print(f"\n[Test {idx+1}] System Role: {role.upper()} | Query: '{q}'")
        
        # Run Legacy
        leg = run_legacy_pipeline(q, role)
        
        # Run Native
        nat = run_native_pipeline(q, role)
        
        results_table.append({
            "Query": q[:30] + "...",
            "Role": role,
            "L_Time": f"{leg['latency']:.3f}s",
            "N_Time": f"{nat['latency']:.3f}s",
            "L_Docs": leg['docs'],
            "N_Docs": nat['docs'],
            "L_Score": f"{leg['top_score']:.2f}",
            "N_Score": f"{nat['top_score']:.2f}"
        })
        
        print(f"  Legacy custom reranker:  {leg['latency']:.3f}s | Output Docs: {leg['docs']}")
        print(f"  Qdrant native ranking:   {nat['latency']:.3f}s | Output Docs: {nat['docs']}")
        
    print("\n\n" + "="*80)
    print(" FINAL COMPARISON REPORT ")
    print("="*80)
    
    try:
        from tabulate import tabulate
        table_data = [[
            r['Query'], r['Role'], r['L_Time'], r['N_Time'], r['L_Docs'], r['N_Docs'], r['L_Score'], r['N_Score']
        ] for r in results_table]
        
        headers = ["Query Snippet", "Role", "Legacy Time", "Native Time", "Legacy Cnt", "Native Cnt", "Legacy Score", "Native Score"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    except ImportError:
        for r in results_table:
            print(r)

if __name__ == "__main__":
    compare()
