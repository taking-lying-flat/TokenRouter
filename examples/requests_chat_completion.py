import requests

url = "http://127.0.0.1:30000/v1/chat/completions"
data = {
    "model": "default",
    "messages": [
        {
            "role": "user",
            "content": "Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her $4$ hours, including $t$ minutes spent in the coffee shop. When she walks at $s + 2$ kilometers per hour, the walk takes her $2$ hours and $24$ minutes, including $t$ minutes spent in the coffee shop. Suppose Aya walks at $s + \\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop.\n",
        }
    ],
    "temperature": 0,
    "top_p": 1,
    "max_tokens": 8192,
}

response = requests.post(url, json=data)
response.raise_for_status()

result = response.json()
print(result["choices"][0]["message"]["content"])
print("usage:", result.get("usage"))

if "usage_details" in result:
    print("usage_details:", result["usage_details"])
