import os
import json
import re
import requests
import uuid

# Import our new optimized RAG module
import improved_and_optimized_RAG as r_app

# ==========================================
# CONFIG
# ==========================================

# Using the NVIDIA NIM model
NVIDIA_API_KEY = r_app.NVIDIA_API_KEY
JUDGE_MODEL        = r_app.LLM_MODEL_NAME
NVIDIA_URL     = "https://integrate.api.nvidia.com/v1/chat/completions"

# ==========================================
# LLM JUDGE PROMPT
# ==========================================

JUDGE_SYSTEM_PROMPT = """You are a strict relevance judge for a RAG system.

Your task is to determine if a retrieved document contains factual information relevant
to answering a given query.

Rules:
- A document is RELEVANT if it directly addresses the query's topic with factual content.
- A document is NOT RELEVANT if it is off-topic.
- We do not care if the document has the exact answer, only if it is topically relevant to the question.

Output ONLY a raw JSON object with NO markdown formatting, NO code fences:
{"is_relevant": true, "reasoning": "One concise sentence explaining why."}"""

# ==========================================
# TEST DATASET FOR KPI EVALUATION
# ==========================================

TEST_CASES = [
    {
        "id": "QA_ROLE_01",
        "role": "admin",
        "query": "How do I wipe up data?",
        "expected_security": "ALLOW",
        "expected_docs_exist": True
    },
    {
        "id": "QA_ROLE_02",
        "role": "volunteer",
        "query": "What are the rules and responsibilities for volunteers?",
        "expected_security": "ALLOW",
        "expected_docs_exist": True
    },
    {
         # A volunteer asking a question about a completely missing topic (Passwords)
        "id": "QA_ROLE_03",
        "role": "volunteer",
        "query": "What are the master passwords for the database?",
        "expected_security": "DENY", # Should technically just return 0 docs via Native threshold 
        "expected_docs_exist": False
    },
    {
        # Volunteer asking an Employee-only question (Wiping Data)
        "id": "QA_ROLE_04",
        "role": "volunteer",
        "query": "How do I wipe up data?",
        "expected_security": "DENY", 
        "expected_docs_exist": True # Docs exist, but Role filters should block them
    },
    {
        "id": "QA_ROLE_05",
        "role": "employee",
        "query": "What are the core duties of a volunteer?",
        "expected_security": "ALLOW", # Employees can read volunteer docs
        "expected_docs_exist": True
    }
]

# ==========================================
# JUDGE CALL
# ==========================================

def judge_relevance(query: str, document: str) -> dict:
    user_message = f"Query: {query}\n\nDocument: {document[:800]}"

    payload = {
        "model": JUDGE_MODEL,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        "temperature": 0.0,
    }
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
    }

    r = requests.post(NVIDIA_URL, json=payload, headers=headers, timeout=30)
    r.raise_for_status()
    raw = r.json()["choices"][0]["message"]["content"].strip()
    raw = re.sub(r"```[a-z]*\n?", "", raw).strip().strip("`")

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        is_rel = "true" in raw.lower()
        return {"is_relevant": is_rel, "reasoning": raw[:120]}


# ==========================================
# METRICS ENGINE
# ==========================================

def run_evaluation():
    print("=" * 80)
    print(" FINAL KPI EVALUATION: OPTIMIZED QDRANT RAG")
    print("=" * 80)
    
    all_precisions = []
    all_recalls = []
    security_passes = 0

    for tc_idx, tc in enumerate(TEST_CASES):
        query   = tc["query"]
        role    = tc["role"]
        
        print(f"\n[TestCase {tc['id']}] Role: {role.upper()} | Query: \"{query}\"")

        # 1. RETRIVE FROM RAG
        # We hook directly into the Qdrant native search func to just get raw docs for KPI evaluation
        raw_hits = r_app.search_qdrant(query, role, return_raw=True)
        
        # 2. EVALUATE SECURITY ACCURACY
        # Security PASS if: Expected DENY & Got 0 Hits, OR Expected ALLOW & Got >0 Hits (where docs actually exist)
        hits_count = len(raw_hits)
        
        sec_pass = False
        if tc["expected_security"] == "DENY" and hits_count == 0:
            sec_pass = True
        elif tc["expected_security"] == "ALLOW" and tc["expected_docs_exist"] and hits_count > 0:
            sec_pass = True
        elif tc["expected_security"] == "DENY" and not tc["expected_docs_exist"] and hits_count == 0:
            sec_pass = True  # Caught by Threshold natively
        elif not tc["expected_docs_exist"] and hits_count == 0:
             sec_pass = True
             
        if sec_pass:
            security_passes += 1
            
        print(f"  -> Security enforcement: {'PASS' if sec_pass else 'FAIL'} (Expected {tc['expected_security']}, Retrieved {hits_count} docs)")
        
        # 3. EVALUATE PRECISION / RECALL
        if hits_count == 0:
             if tc["expected_docs_exist"] and tc["expected_security"] == "ALLOW":
                 # We should have found something, recall is 0
                 all_precisions.append(1.0) # Nothing returned = precision isn't ruined
                 all_recalls.append(0.0)    # Nothing returned = recall failed
             else:
                 # We shouldn't have found anything anyway
                 all_precisions.append(1.0)
                 all_recalls.append(1.0)
        else:
             relevant_found = 0
             for rank, doc in enumerate(raw_hits, 1):
                 try:
                     judge_res = judge_relevance(query, doc['content'])
                     is_rel = judge_res.get("is_relevant", False)
                 except:
                     is_rel = False
                     
                 if is_rel: relevant_found += 1
             
             precision = relevant_found / hits_count
             # Simple binary recall estimation: If we found >0 relevant documents out of the >0 retrieved, recall = 1.0 (We got the answer to the LLM)
             recall = 1.0 if relevant_found > 0 else 0.0
             
             all_precisions.append(precision)
             all_recalls.append(recall)
             print(f"  -> Semantic Precision: {precision*100:.1f}%")

    # Aggregate
    avg_p = sum(all_precisions) / len(all_precisions)
    avg_r = sum(all_recalls) / len(all_recalls)
    avg_f1 = (2 * avg_p * avg_r) / (avg_p + avg_r) if (avg_p + avg_r) > 0 else 0.0
    sec_acc = security_passes / len(TEST_CASES)

    print("\n" + "=" * 80)
    print(" FINAL RAG METRICS (Optimized Qdrant)")
    print("=" * 80)
    print(f" Security Accuracy    : {sec_acc*100:.1f}%  (Role Constraints & Thresholds)")
    print(f" Semantic Precision   : {avg_p*100:.1f}%  (Quality of Docs passed to LLM)")
    print(f" Semantic Recall      : {avg_r*100:.1f}%  (Success finding the answer)")
    print(f" Overall Quality F1   : {avg_f1:.3f}")
    print("=" * 80)

if __name__ == "__main__":
    run_evaluation()
