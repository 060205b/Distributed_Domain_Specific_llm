from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer
import torch  # Import torch
from fastapi.staticfiles import StaticFiles

# --- Model Loading (ONNX Runtime) ---
model_path = "onnx_model"  # Path to your ONNX model directory
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = ORTModelForCausalLM.from_pretrained(model_path)


# --- FastAPI Setup ---
app = FastAPI()

class Query(BaseModel):
    text: str

@app.post("/chat")
async def chat(query: Query):
    try:
        inputs = tokenizer(query.text, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=50)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Serve Static Files (Web Interface) ---
app.mount("/", StaticFiles(directory="static", html=True), name="static")
