from fastapi import FastAPI
from pydantic import BaseModel
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch

# Model setup
MODEL_NAME = "facebook/m2m100_418M"  # smaller, CPU-friendly
tokenizer = M2M100Tokenizer.from_pretrained(MODEL_NAME)
model = M2M100ForConditionalGeneration.from_pretrained(MODEL_NAME)
device = torch.device("cpu")  # force CPU
model.to(device)

# FastAPI setup
app = FastAPI()

class TranslationRequest(BaseModel):
    text: str
    source_lang: str
    target_lang: str

@app.get("/")
def root():
    return {"message": "M2M-100 Translator API running 🚀", "endpoints": ["/translate", "/health"]}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/translate")
def translate(req: TranslationRequest):
    tokenizer.src_lang = req.source_lang
    inputs = tokenizer(req.text, return_tensors="pt").to(device)

    generated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.get_lang_id(req.target_lang)
    )

    translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return {"translation": translation[0]}
