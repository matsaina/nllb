from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global variables for model and tokenizer
model = None
tokenizer = None

# Language code mapping (NLLB uses different codes)
LANGUAGE_CODES = {
    'en': 'eng_Latn',
    'es': 'spa_Latn',
    'fr': 'fra_Latn',
    'de': 'deu_Latn',
    'it': 'ita_Latn',
    'pt': 'por_Latn',
    'ru': 'rus_Cyrl',
    'zh': 'zho_Hans',
    'ja': 'jpn_Jpan',
    'ko': 'kor_Hang',
    'ar': 'arb_Arab',
    'hi': 'hin_Deva',
    'tr': 'tur_Latn',
    'pl': 'pol_Latn',
    'nl': 'nld_Latn',
    'sv': 'swe_Latn',
    'da': 'dan_Latn',
    'no': 'nob_Latn',
    'fi': 'fin_Latn',
    'el': 'ell_Grek',
    'he': 'heb_Hebr',
    'th': 'tha_Thai',
    'vi': 'vie_Latn',
    'id': 'ind_Latn',
    'ms': 'zsm_Latn',
    'tl': 'tgl_Latn',
    'rw': 'kin_Latn',
    'sw': 'swh_Latn'
}

def load_model():
    global model, tokenizer
    try:
        logger.info("Loading NLLB model...")
        model_name = "facebook/nllb-200-distilled-600M"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            model = model.cuda()
            logger.info("Model loaded on GPU")
        else:
            logger.info("Model loaded on CPU")
            
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'gpu_available': torch.cuda.is_available()
    })

@app.route('/translate', methods=['POST'])
def translate():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        text = data.get('text', '')
        source_lang = data.get('source_lang', 'en')
        target_lang = data.get('target_lang', 'es')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
            
        # Convert language codes to NLLB format
        src_lang_code = LANGUAGE_CODES.get(source_lang, source_lang)
        tgt_lang_code = LANGUAGE_CODES.get(target_lang, target_lang)
        
        # Tokenize and translate
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate translation
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang_code],
                max_length=512,
                num_beams=5,
                early_stopping=True
            )
        
        # Decode the translation
        translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        
        return jsonify({
            'translation': translation,
            'source_lang': source_lang,
            'target_lang': target_lang,
            'original_text': text
        })
        
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/languages', methods=['GET'])
def get_supported_languages():
    return jsonify({
        'supported_languages': list(LANGUAGE_CODES.keys()),
        'language_mapping': LANGUAGE_CODES
    })

if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=8000, debug=False)