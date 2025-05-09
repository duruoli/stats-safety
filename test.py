import openai
import os


openai.api_key = ""

try:
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": "Hello!"}
        ]
    )
    print("✅ API key works!")
    print("Response:", response['choices'][0]['message']['content'])
except Exception as e:
    print("❌ Something went wrong:", e)
