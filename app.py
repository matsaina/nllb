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

# Language code mapping for NLLB
LANGUAGE_CODES = {
    'english': 'eng_Latn',
    'swahili': 'swh_Latn',
    'kinyarwanda': 'kin_Latn',
    'french': 'fra_Latn',
    'spanish': 'spa_Latn',
    'german': 'deu_Latn',
    'portuguese': 'por_Latn',
    'arabic': 'arb_Arab',
    'chinese': 'zho_Hans',
    'japanese': 'jpn_Jpan',
    'korean': 'kor_Hang',
    'hindi': 'hin_Deva'
}

# Global variables for model and tokenizer
model = None
tokenizer = None

def load_model():
    """Load the NLLB model and tokenizer"""
    global model, tokenizer
    
    try:
        logger.info("Loading NLLB-200-distilled-600M model...")
        model_name = "facebook/nllb-200-distilled-600M"
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Use GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        logger.info(f"Model loaded successfully on {device}")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

def translate_text(text, src_lang, tgt_lang):
    """Translate text from source language to target language"""
    try:
        # Get language codes
        src_code = LANGUAGE_CODES.get(src_lang.lower())
        tgt_code = LANGUAGE_CODES.get(tgt_lang.lower())
        
        if not src_code or not tgt_code:
            raise ValueError(f"Unsupported language pair: {src_lang} -> {tgt_lang}")
        
        # Tokenize input
        tokenizer.src_lang = src_code
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Move inputs to same device as model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate translation
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.lang_code_to_id[tgt_code],
                max_length=512,
                num_beams=5,
                early_stopping=True
            )
        
        # Decode translation
        translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return translation
        
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        raise

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/languages', methods=['GET'])
def get_supported_languages():
    """Get list of supported languages"""
    return jsonify({
        'supported_languages': list(LANGUAGE_CODES.keys())
    })

@app.route('/translate', methods=['POST'])
def translate():
    """Main translation endpoint"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        text = data.get('text', '').strip()
        src_lang = data.get('source_language', '').strip()
        tgt_lang = data.get('target_language', '').strip()
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        if not src_lang or not tgt_lang:
            return jsonify({'error': 'Source and target languages are required'}), 400
        
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Perform translation
        translation = translate_text(text, src_lang, tgt_lang)
        
        return jsonify({
            'original_text': text,
            'translated_text': translation,
            'source_language': src_lang,
            'target_language': tgt_lang
        })
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Translation endpoint error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Load model on startup
    if load_model():
        logger.info("Starting Flask application...")
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        logger.error("Failed to load model. Exiting.")
        exit(1)
