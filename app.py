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

from transformers import pipeline

translator = pipeline(
    "translation",
    model="facebook/nllb-200-distilled-600M",
    tokenizer="facebook/nllb-200-distilled-600M",
    device=-1  # CPU only; change to 0 if using GPU
)

@app.post("/translate")
def translate(req: TranslationRequest):
    # Use pipeline translation
    result = translator(
        req.text,
        src_lang=req.source_lang,
        tgt_lang=req.target_lang
    )
    return {"translation": result[0]['translation_text']}



@app.get("/")
def root():
    return {"message": "NLLB Translator API is running 🚀", "endpoints": ["/translate", "/health"]}

@app.get("/health")
def health():
    return {"status": "ok"}
