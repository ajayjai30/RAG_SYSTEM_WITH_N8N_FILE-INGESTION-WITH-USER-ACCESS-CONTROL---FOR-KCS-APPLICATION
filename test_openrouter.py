import requests
import json

key = "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"

models_to_test = [
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "qwen/qwen-2.5-coder-32b-instruct:free",
    "google/gemini-2.0-flash-lite-preview-02-05:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "mistralai/mistral-nemo:free"
]

payload = {
    "messages": [{"role": "user", "content": "hi"}],
    "tools": [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    }]
}

for model in models_to_test:
    payload["model"] = model
    print(f"Testing {model}...")
    try:
        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {key}"}
        )
        if r.status_code == 200:
            print(f"✅ SUCCESS: {model} supports tools!")
            break
        else:
            print(f"❌ FAILED ({r.status_code}): {r.text}")
    except Exception as e:
        print(f"Error: {e}")
