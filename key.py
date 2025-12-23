import os

key = os.getenv("GROQ_API_KEY")

print("GROQ_API_KEY:", key)
print("Key loaded:", bool(key))
