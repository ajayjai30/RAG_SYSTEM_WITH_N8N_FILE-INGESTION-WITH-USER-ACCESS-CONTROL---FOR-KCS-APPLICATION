import os
import uuid
import time
from typing import List

# Import our optimized Agentic RAG module
import improved_and_optimized_RAG as rag

# ==========================================
#  TEST CONFIGURATION
# ==========================================

print("=" * 80)
print(" AGENTIC TOOL CALLING & SECURITY EVALUATION")
print("=" * 80)
print("This script verifies that the LangGraph ReAct agent correctly 'thinks'")
print("and invokes the 'search_internal_database' tool for factual questions,")
print("while correctly strictly refusing general knowledge questions.")
print("=" * 80)

# ==========================================
#  TEST CASES
# ==========================================

TEST_CASES = [
    {
        "id": "TOOL_01",
        "type": "Internal Policy (Tool Expected)",
        "question": "What is the procedure for handling a loose animal?",
        "role": "volunteer",
        "expect_tool_call": True,
        "expect_deny_response": False
    },
    {
        "id": "TOOL_02",
        "type": "Internal Data - Role Restricted",
        "question": "What is the hourly wage for employees?",
        "role": "volunteer",
        "expect_tool_call": True, # It should STILL call the tool, but the tool will return ACCESS DENIED
        "expect_deny_response": True # Therefore the final answer should be a refusal
    },
    {
        "id": "TOOL_03",
        "type": "General Knowledge (No Tool Expected)",
        "question": "What is the capital of France?",
        "role": "admin",
        "expect_tool_call": False,
        "expect_deny_response": True # System prompt forces refusal
    },
    {
        "id": "TOOL_04",
        "type": "Small Talk (No Tool Expected)",
        "question": "Hello! Write me a python script to sort an array.",
        "role": "employee",
        "expect_tool_call": False,
        "expect_deny_response": True # System prompt forces refusal
    }
]

# ==========================================
#  EVALUATION ENGINE
# ==========================================

def run_evaluation():
    success_count = 0
    total_tests = len(TEST_CASES)
    
    for idx, tc in enumerate(TEST_CASES):
        print(f"\n[TestCase {tc['id']}] {tc['type']}")
        print(f" > Role: {tc['role'].upper()}")
        print(f" > Question: \"{tc['question']}\"")
        
        session_id = str(uuid.uuid4())
        
        # We hook into process_query. The returned 'debug_output' 
        # (retrieved_docs) contains the tool's raw output.
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
        
        # 1. Did it call the tool?
        # In our implementation, `retrieved_docs` is populated ONLY IF the tool was called.
        did_call_tool = bool(retrieved_docs.strip())
        
        # Special check: If it hit the tool but the tool returned the backend denial string,
        # retrieved_docs might be empty in the UI depending on how we handle it, but 
        # let's look at the bot's final answer to see if it mentions clearance.
        is_denied = "strict internal company assistant" in bot_answer.lower() or \
                    "access to that information" in bot_answer.lower() or \
                    "database blocked" in bot_answer.lower() or \
                    "clearance" in bot_answer.lower()
                    
        print("\n--- Agent Trace ---")
        if did_call_tool:
            print(" [PASS] Tool Called: 'search_internal_database'")
            if tc['expect_tool_call']:
                print("    (Expected behavior)")
            else:
                print("    [FAIL] Agent hallucinated a tool call for general knowledge.")
        else:
            print(" [INFO] Tool NOT Called.")
            if not tc['expect_tool_call']:
                print("    (Expected behavior: Strict off-topic refusal)")
            else:
                print("    [FAIL] Agent hallucinated knowledge instead of searching.")
                
        print(f" \n--- Final Answer ---\n{bot_answer.strip()}")
        
        # 2. Evaluate Pass/Fail
        test_pass = False
        
        if tc['expect_tool_call'] == did_call_tool:
             if tc['expect_deny_response']:
                 if is_denied:
                     test_pass = True
                     print("\n RESULT: PASS (Correct Tool Usage & Correct Refusal)")
                 else:
                     print("\n RESULT: FAIL (Failed to generate security refusal/off-topic refusal)")
             else:
                 if not is_denied:
                     test_pass = True
                     print("\n RESULT: PASS (Correct Tool Usage & Answered Question)")
                 else:
                     print("\n RESULT: FAIL (Refused to answer when it shouldn't have)")
        else:
             print("\n RESULT: FAIL (Tool calling behavior did not match expectations)")
             
        if test_pass:
            success_count += 1
            
        print("-" * 60)
        
        # Prevent hitting OpenRouter Free Tier burst rate limits
        if idx < total_tests - 1:
            print("Sleeping for 10 seconds to avoid upstream rate limits...")
            time.sleep(10)

    print("\n" + "=" * 80)
    print(f" FINAL SCORE: {success_count}/{total_tests} Tests Passed")
    print("=" * 80)
    
if __name__ == "__main__":
    # Ensure NVIDIA key is set in rag module
    if not rag.NVIDIA_API_KEY or "nvapi-" not in rag.NVIDIA_API_KEY:
        print("WARNING: NVIDIA API key missing or invalid in improved_and_optimized_RAG.py")
        
    try:
        run_evaluation()
    except Exception as e:
        print(f"\nEvaluation failed with error: {e}")
        if "429" in str(e):
             print("\n=> OPENROUTER API RATE LIMIT EXCEEDED. Please try again later or upgrade tier.")
