from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="NLLB-200 Translation API")

# Define request model
class TranslateRequest(BaseModel):
    text: str
    source_lang: str  # e.g., "eng_Latn"
    target_lang: str  # e.g., "fra_Latn"

# Initialize translation pipeline
translator = pipeline(
    "translation",
    model="facebook/nllb-200-distilled-1.3B",
)

@app.post("/translate")
def translate(req: TranslateRequest):
    try:
        result = translator(
            req.text,
            src_lang=req.source_lang,
            tgt_lang=req.target_lang
        )
        return {"translation": result[0]["translation_text"]}
    except Exception as e:
        return {"error": str(e)}
