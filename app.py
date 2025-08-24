from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

app = FastAPI()

class TranslationRequest(BaseModel):
    text: str
    source_lang: str
    target_lang: str

@app.post("/translate")
def translate(req: TranslationRequest):
    inputs = tokenizer(
        req.text,
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id[req.target_lang]
    )
    translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
    return {"translation": translated_text[0]}

@app.get("/")
def root():
    return {"message": "NLLB Translator API is running 🚀", "endpoints": ["/translate", "/health"]}

@app.get("/health")
def health():
    return {"status": "ok"}
