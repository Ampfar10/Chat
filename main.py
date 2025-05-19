import json
import os
import threading
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

MEMORY_FILE = "memory.json"
DATA_FILE = "conversations.json"
MODEL_DIR = "trained_model"

# Load or create memory
memory = {}
if os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE, "r") as f:
        memory = json.load(f)

# Load or create conversations
conversations = []
if os.path.exists(DATA_FILE):
    with open(DATA_FILE, "r") as f:
        conversations = json.load(f)

# Load model
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR if os.path.exists(MODEL_DIR) else "distilgpt2")
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR if os.path.exists(MODEL_DIR) else "distilgpt2")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

def save_memory():
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=4)

def save_conversations():
    with open(DATA_FILE, "w") as f:
        json.dump(conversations, f, indent=4)

def remember_facts(text):
    if "my name is" in text.lower():
        name = text.lower().split("my name is")[-1].strip().split()[0]
        memory["name"] = name
        save_memory()
        return f"Nice to meet you, {name}!"
    return None

def find_exact_response(user_input):
    for pair in conversations:
        if pair["input"] == user_input:
            return pair["response"]
    return None

def generate_response(prompt):
    result = generator(prompt, max_length=60, do_sample=True, temperature=0.7)
    return result[0]['generated_text'].replace(prompt, '').strip()

def background_retrain():
    while True:
        os.system("python train.py")
        time.sleep(600)  # retrain every 10 minutes

# Start retraining thread
threading.Thread(target=background_retrain, daemon=True).start()

print("Bot: Hello! I'm your learning bot. Type 'exit' to quit.")

while True:
    user = input("You: ").strip()
    if user == "exit":
        print("Bot: Bye!")
        break

    fact_response = remember_facts(user)
    if fact_response:
        print("Bot:", fact_response)
        continue

    known = find_exact_response(user)
    if known:
        print("Bot:", known)
    else:
        print("Bot: I don't know how to respond. How should I reply?")
        teacher = input("You (teach me): ").strip()
        conversations.append({"input": user, "response": teacher})
        save_conversations()
        print("Bot: Thanks! I'll remember that.")
