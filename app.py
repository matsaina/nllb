from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = FastAPI(title="NLLB Translator API 🚀")

MODEL_NAME = "facebook/nllb-200-1.3B"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

class TranslateRequest(BaseModel):
    text: str
    source_lang: str
    target_lang: str

@app.get("/")
def root():
    return {"message": "NLLB Translator API is running 🚀", "endpoints": ["/translate"]}

@app.post("/translate")
def translate(req: TranslateRequest):
    # Tokenize input text
    inputs = tokenizer(req.text, return_tensors="pt")
    
    # Generate translation
    generated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id[req.target_lang],
        max_length=256
    )
    
    translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return {"translation": translation}
