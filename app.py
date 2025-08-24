from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = FastAPI()

MODEL_NAME = "facebook/nllb-200-distilled-1.3B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

class TranslateRequest(BaseModel):
    text: str
    source_lang: str  # e.g., "eng_Latn"
    target_lang: str  # e.g., "kin_Latn"

@app.post("/translate")
def translate(req: TranslateRequest):
    # Tokenize input normally
    inputs = tokenizer(req.text, return_tensors="pt")

    # Get the target language ID from the tokenizer
    target_lang_id = tokenizer.get_lang_id(req.target_lang)

    # Generate translation
    generated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=target_lang_id,
        decoder_start_token_id=target_lang_id
    )

    translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return {"translation": translation}
