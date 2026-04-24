from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from utils import find_nearby_doctors
import os
import uvicorn

app = FastAPI(title="Healthcare Chatbot")

# Serve the static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ==========================================
# 1. Model Setup
# ==========================================
MODEL_ID = os.environ.get("MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
print(f"Loading Model: {MODEL_ID}")

device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True
    )
    if device == "cpu":
        model = model.to("cpu")

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    MODEL_LOADED = True
except Exception as e:
    print(f"Error loading model: {e}")
    MODEL_LOADED = False

# ==========================================
# 2. Endpoints
# ==========================================

class ChatRequest(BaseModel):
    message: str

@app.get("/", response_class=HTMLResponse)
async def read_root():
    # Serve the main index.html file
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    user_message = request.message
    
    # 1. Check if the user is asking for a doctor location
    if "doctor near" in user_message.lower() or "clinic near" in user_message.lower() or "hospital near" in user_message.lower():
        words = user_message.lower().split("near")
        if len(words) > 1:
            location = words[1].strip(" ?.!\"'")
            if location:
                return {"reply": find_nearby_doctors(location)}
            else:
                return {"reply": "Please specify the city or zip code where you want to find a doctor."}

    # 2. Otherwise, use the Local Language Model
    if not MODEL_LOADED:
        return {"reply": "I'm sorry, the AI model is currently offline or failed to load. Please try again later."}

    prompt = f"<|system|>\nYou are a helpful medical assistant. However, you are not a replacement for a doctor. Always recommend seeing a professional.\n<|user|>\n{user_message}\n<|assistant|>\n"
    
    try:
        outputs = pipe(prompt)
        response_text = outputs[0]["generated_text"]
        reply = response_text.split("<|assistant|>\n")[-1].strip()
        return {"reply": reply}
    except Exception as e:
        return {"reply": f"An error occurred while generating a response: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)