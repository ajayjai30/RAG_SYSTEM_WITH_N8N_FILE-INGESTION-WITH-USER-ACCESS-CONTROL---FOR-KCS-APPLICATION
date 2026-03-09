import requests
key = "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"
url = "https://integrate.api.nvidia.com/v1/models"
headers = {"Authorization": f"Bearer {key}", "Accept": "application/json"}
try:
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        models = [m['id'] for m in r.json().get('data', [])]
        print("Models:", models[:15])
    else:
        print("Failed to get models:", r.status_code, r.text)
except Exception as e:
    print("Error:", e)
