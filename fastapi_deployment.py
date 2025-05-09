from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import Response
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import pickle
import os
import logging
from prometheus_client import Counter, Histogram
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ml-service")

# Setup metrics
TRANSLATION_REQUESTS = Counter("translation_requests_total", "Total number of translation requests")
TRANSLATION_ERRORS = Counter("translation_errors_total", "Total number of translation errors")
TRANSLATION_LATENCY = Histogram("translation_latency_seconds", "Translation request latency in seconds")

app = FastAPI(title="ML Translation Model Service")

# Model paths from environment variables with defaults pointing to latest model
MODEL_DIR = os.getenv("MODEL_DIR", "models/itsekiri_translator_20250507_215757")
ENCODER_PATH = os.getenv("ENCODER_MODEL_PATH", os.path.join(MODEL_DIR, "encoder.weights.h5"))
DECODER_PATH = os.getenv("DECODER_MODEL_PATH", os.path.join(MODEL_DIR, "decoder.weights.h5"))
INP_TOKENIZER_PATH = os.getenv("INP_TOKENIZER_PATH", os.path.join(MODEL_DIR, "inp_tokenizer.json"))
TARG_TOKENIZER_PATH = os.getenv("TARG_TOKENIZER_PATH", os.path.join(MODEL_DIR, "targ_tokenizer.json"))

# Model loading with lazy initialization
class ModelLoader:
    def __init__(self):
        self.encoder = None
        self.decoder = None
        self.inp_tokenizer = None
        self.targ_tokenizer = None
        self.is_loaded = False

    def load_models(self):
        if not self.is_loaded:
            try:
                logger.info("Loading ML models and tokenizers...")
                
                # Load tokenizers first to get vocabulary sizes
                with open(INP_TOKENIZER_PATH, "r", encoding='utf-8') as f:
                    inp_tokenizer_data = f.read()
                    self.inp_tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(inp_tokenizer_data)
                
                with open(TARG_TOKENIZER_PATH, "r", encoding='utf-8') as f:
                    targ_tokenizer_data = f.read()
                    self.targ_tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(targ_tokenizer_data)
                
                # Get vocabulary sizes from tokenizers
                self.input_vocab_size = len(self.inp_tokenizer.word_index) + 1
                self.target_vocab_size = len(self.targ_tokenizer.word_index) + 1
                
                logger.info(f"Loaded tokenizers. Input vocab size: {self.input_vocab_size}, Target vocab size: {self.target_vocab_size}")
                
                # Now initialize models with correct vocabulary sizes
                self.encoder = tf.keras.Sequential([
                    tf.keras.layers.Embedding(self.input_vocab_size, 256),
                    tf.keras.layers.GRU(256, return_sequences=True, return_state=True)
                ])
                
                self.decoder = tf.keras.Sequential([
                    tf.keras.layers.Embedding(self.target_vocab_size, 256),
                    tf.keras.layers.GRU(256, return_sequences=True, return_state=True),
                    tf.keras.layers.Dense(self.target_vocab_size)
                ])
                
                # Load pre-trained weights
                self.encoder.load_weights(ENCODER_PATH)
                self.decoder.load_weights(DECODER_PATH)
                
                self.is_loaded = True
                logger.info("Models loaded successfully with correct vocabulary sizes")
            except Exception as e:
                logger.error(f"Failed to load models: {str(e)}")
                raise RuntimeError(f"Failed to load models: {str(e)}")
        
        return self

model_loader = ModelLoader()

# Dependency to ensure models are loaded
async def get_models():
    return model_loader.load_models()

# Define input and output schemas
class TranslationRequest(BaseModel):
    itsekiri_sentence: str
    model_version: str = "default"  # Allow for model versioning

class TranslationResponse(BaseModel):
    english_translation: str
    processing_time: float
    model_version: str

def translate_text(sentence, loader):
    try:
        logger.info(f"Input sentence: '{sentence}'")
        
        # Preprocess and tokenize
        sentence_seq = loader.inp_tokenizer.texts_to_sequences([sentence])
        logger.info(f"Tokenized sequence: {sentence_seq}")
        
        # Check if any tokens were found
        if not any(sentence_seq[0]):
            logger.warning("No tokens found in input text - word may not be in vocabulary")
            return ""
            
        sentence_padded = tf.keras.preprocessing.sequence.pad_sequences(sentence_seq, maxlen=20, padding='post')
        logger.info(f"Padded sequence: {sentence_padded}")

        result = ''
        hidden = [tf.zeros((1, 256))]  # Using 256 units to match training config
        
        # Generate encoder output
        enc_out, enc_hidden = loader.encoder(tf.convert_to_tensor(sentence_padded), hidden)
        logger.debug(f"Encoder output shape: {enc_out.shape}, hidden state shape: {enc_hidden.shape}")

        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([loader.targ_tokenizer.word_index.get('<start>', 1)], 0)

        for t in range(20):  # Max output length
            predictions, dec_hidden = loader.decoder(dec_input, dec_hidden)
            predicted_id = tf.argmax(predictions[0][0]).numpy()
            next_word = loader.targ_tokenizer.index_word.get(predicted_id, '')
            
            logger.debug(f"Step {t}: predicted_id={predicted_id}, word='{next_word}'")

            if next_word == '<end>':
                break
            if next_word and next_word not in ['<start>', '<unk>']:
                result += next_word + ' '
            
            dec_input = tf.expand_dims([predicted_id], 0)

        final_result = result.strip()
        logger.info(f"Final translation: '{final_result}'")
        return final_result

    except Exception as e:
        logger.error(f"Translation error: {str(e)}", exc_info=True)
        raise

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model_loader.is_loaded}

# Define the prediction endpoint
@app.post("/translate", response_model=TranslationResponse)
async def translate(request: TranslationRequest, models=Depends(get_models)):
    TRANSLATION_REQUESTS.inc()
    start_time = time.time()
    
    try:
        # Log the request
        logger.info(f"Processing translation request: {request.itsekiri_sentence[:50]}...")
        
        # Process the translation
        translation = translate_text(request.itsekiri_sentence, models)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        TRANSLATION_LATENCY.observe(processing_time)
        
        # Return the response
        return TranslationResponse(
            english_translation=translation,
            processing_time=processing_time,
            model_version=request.model_version
        )
    except Exception as e:
        TRANSLATION_ERRORS.inc()
        logger.error(f"Translation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Model information endpoint
@app.get("/model/info")
async def model_info(models=Depends(get_models)):
    return {
        "encoder_path": ENCODER_PATH,
        "decoder_path": DECODER_PATH,
        "is_loaded": models.is_loaded
    }

# Metrics endpoint for Prometheus
@app.get("/metrics")
async def metrics():
    from prometheus_client import generate_latest
    return Response(content=generate_latest(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)