import json
import os
import uuid
from typing import List, Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Import the main RAG processing logic
import improved_and_optimized_RAG as rag

# ==========================================
#  EVALUATOR CONFIGURATION 
# ==========================================

# Re-use the NVIDIA config for our "LLM-as-a-judge"
JUDGE_MODEL_NAME = "openai/gpt-oss-20b" 

judge_llm = ChatOpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=rag.NVIDIA_API_KEY,
    model=JUDGE_MODEL_NAME,
    temperature=0.0 # Strict evaluation
)

# --- Prompts for Evaluation ---

FAITHFULNESS_PROMPT = """You are an expert evaluator. 
Your task is to determine if the generated 'Answer' is derived ONLY from the provided 'Context'.
If the answer contains specific claims, facts, or details that are NOT present in the Context, it is unfaithful.
Note: If the answer is a generic refusal like "I don't have access to that information" or a predefined security denial, and the context is empty, it is considered Faithful.

Context:
{context}

Answer:
{answer}

Output exactly "1" if faithful, or "0" if unfaithful. Do NOT output any other text or explanation.
"""

RELEVANCE_PROMPT = """You are an expert evaluator.
Your task is to determine if the 'Answer' directly addresses the user's 'Question'.
If the answer is relevant, or if it correctly states it cannot answer due to lack of information/security, it is considered relevant.
If it rambles about an unrelated topic, it is not relevant.

Question:
{question}

Answer:
{answer}

Output exactly "1" if relevant, or "0" if not relevant. Do NOT output any other text or explanation.
"""

faithfulness_chain = ChatPromptTemplate.from_template(FAITHFULNESS_PROMPT) | judge_llm | StrOutputParser()
relevance_chain = ChatPromptTemplate.from_template(RELEVANCE_PROMPT) | judge_llm | StrOutputParser()

# ==========================================
#  TEST DATASET
# ==========================================
# We define assertions based on whether we expect the system to ALLOW or DENY the context based on roles.
TEST_CASES = [
    {
        "id": "TC01",
        "question": "What is the hourly wage for employees?",
        "role": "volunteer",
        "expected_access": "DENY" # Volunteers shouldn't see employee data
    },
    {
        "id": "TC02",
        "question": "What are the core duties of a volunteer?",
        "role": "volunteer",
        "expected_access": "ALLOW"
    },
    {
        "id": "TC03",
        "question": "What are the core duties of a volunteer?",
        "role": "employee",
        "expected_access": "ALLOW" # Employees should see volunteer data
    },
    {
        "id": "TC04",
        "question": "What is the hourly wage for employees?",
        "role": "admin",
        "expected_access": "ALLOW" # Admins see everything
    },
    {
        "id": "TC05",
        "question": "What is the procedure for handling a loose animal?",
        "role": "volunteer",
        "expected_access": "ALLOW"
    }
]

# ==========================================
#  EVALUATION ENGINE
# ==========================================

def extract_binary_score(llm_output: str) -> int:
    """Safely parse the LLM's binary response, handling potential thinking tokens or extra whitespace."""
    clean_text = llm_output.strip()
    if "1" in clean_text.split()[-1] or "1" == clean_text:
        return 1
    elif "0" in clean_text.split()[-1] or "0" == clean_text:
        return 0
    else:
        # Default fallback if parsing fails
        return 0

def run_evaluation():
    print(f"Starting RAG Evaluation Engine with {len(TEST_CASES)} cases...")
    print("-" * 80)
    
    results = []
    
    for idx, tc in enumerate(TEST_CASES):
        print(f"Running Test {idx+1}/{len(TEST_CASES)}: [Role: {tc['role']}] {tc['question']}")
        
        session_id = str(uuid.uuid4())
        
        # 1. Execute Query against our RAG system
        history, retrieved_docs, _, _ = rag.process_query(
            message=tc['question'],
            history=None,
            user_role=tc['role'],
            session_id=session_id
        )
        
        bot_answer = ""
        if history:
            last_msg = history[-1]
            if isinstance(last_msg, dict):
                bot_answer = last_msg.get('content', '')
            elif isinstance(last_msg, (list, tuple)):
                bot_answer = last_msg[1]
        
        # 2. Security / Access Evaluation
        actual_access = "DENY" if "ACCESS DENIED" in bot_answer else "ALLOW"
        security_pass = actual_access == tc['expected_access']
        
        # 3. Quality Evaluation (Faithfulness & Relevance) using LLM
        faithfulness_score = 0
        relevance_score = 0
        
        # We only evaluate Faithfulness if there was context, otherwise we assume a DENY response is purely based on the programmatic denial logic.
        if "ACCESS DENIED" in bot_answer and not retrieved_docs:
             faithfulness_score = 1
             relevance_score = 1
        else:
             faithfulness_raw = faithfulness_chain.invoke({"context": retrieved_docs, "answer": bot_answer})
             faithfulness_score = extract_binary_score(faithfulness_raw)
             
             relevance_raw = relevance_chain.invoke({"question": tc['question'], "answer": bot_answer})
             relevance_score = extract_binary_score(relevance_raw)
             
        # Record results
        results.append({
            "ID": tc['id'],
            "Role": tc['role'],
            "Security": "✅ PASS" if security_pass else "❌ FAIL",
            "Faithful": "✅ PASS" if faithfulness_score else "❌ FAIL",
            "Relevant": "✅ PASS" if relevance_score else "❌ FAIL",
            "Context": retrieved_docs[:100] + "..." if retrieved_docs else "None",
            "Answer": bot_answer[:100] + "...",
        })
        print(f"  -> Security: {security_pass} | Faithful: {bool(faithfulness_score)} | Relevant: {bool(relevance_score)}")
        
    # Print Final Report
    print("\n\n" + "="*80)
    print(" " * 25 + " FINAL EVALUATION REPORT")
    print("="*80)
    
    try:
        from tabulate import tabulate
        
        # Format for tabulate
        table_data = [[
            r['ID'], r['Role'], r['Security'], r['Faithful'], r['Relevant']
        ] for r in results]
        
        headers = ["ID", "Role", "Security (Access)", "Faithfulness", "Relevance"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    except ImportError:
        # Fallback if tabulate isn't installed
        for r in results:
            print(f"[{r['ID']}] Role: {r['Role']:<10} | Sec: {r['Security']} | Faith: {r['Faithful']} | Rel: {r['Relevant']}")
            
    print("="*80)
    
if __name__ == "__main__":
    run_evaluation()
