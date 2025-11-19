# backend.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_PATH = "chatbot_model"   # folder you showed in the screenshot

app = FastAPI()

# Load model & tokenizer only once
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

class ChatRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 150
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True


@app.post("/generate")
def generate_text(request: ChatRequest):
    inputs = tokenizer(
        request.prompt,
        return_tensors="pt"
    ).to(device)

    output = model.generate(
        **inputs,
        max_new_tokens=request.max_new_tokens,
        do_sample=request.do_sample,
        temperature=request.temperature,
        top_p=request.top_p,
    )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return {"response": text}
