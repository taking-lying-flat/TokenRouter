import os

from openai import OpenAI

client = OpenAI(
    base_url=os.environ.get("TOKENROUTING_BASE_URL", "http://127.0.0.1:30000/v1"),
    api_key=os.environ.get("TOKENROUTING_API_KEY", "EMPTY"),
)

response = client.chat.completions.create(
    model="default",
    messages=[
        {
            "role": "user",
            "content": "Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her $4$ hours, including $t$ minutes spent in the coffee shop. When she walks at $s + 2$ kilometers per hour, the walk takes her $2$ hours and $24$ minutes, including $t$ minutes spent in the coffee shop. Suppose Aya walks at $s + \\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop.\n",
        }
    ],
    temperature=0,
    top_p=1,
    max_tokens=8192,
)

print(response.choices[0].message.content)
print("usage:", response.usage)

usage_details = getattr(response, "usage_details", None)
if usage_details is not None:
    print("usage_details:", usage_details)
