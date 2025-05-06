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

# Model paths from environment variables
ENCODER_PATH = os.getenv("ENCODER_MODEL_PATH", "encoder_model_path")
DECODER_PATH = os.getenv("DECODER_MODEL_PATH", "decoder_model_path")
INP_TOKENIZER_PATH = os.getenv("INP_TOKENIZER_PATH", "inp_lang_tokenizer.pkl")
TARG_TOKENIZER_PATH = os.getenv("TARG_TOKENIZER_PATH", "targ_lang_tokenizer.pkl")

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
                self.encoder = tf.keras.models.load_model(ENCODER_PATH)
                self.decoder = tf.keras.models.load_model(DECODER_PATH)
                
                with open(INP_TOKENIZER_PATH, "rb") as f:
                    self.inp_tokenizer = pickle.load(f)
                
                with open(TARG_TOKENIZER_PATH, "rb") as f:
                    self.targ_tokenizer = pickle.load(f)
                
                self.is_loaded = True
                logger.info("Models and tokenizers loaded successfully")
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
    sentence_seq = loader.inp_tokenizer.texts_to_sequences([sentence])
    sentence_padded = tf.keras.preprocessing.sequence.pad_sequences(sentence_seq, maxlen=20, padding='post')

    result = ''
    hidden = [tf.zeros((1, 512))]
    enc_out, enc_hidden = loader.encoder(tf.convert_to_tensor(sentence_padded), hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([loader.targ_tokenizer.word_index['<start>']], 0)

    for _ in range(20):  # Max output length
        predictions, dec_hidden = loader.decoder(dec_input, dec_hidden)
        predicted_id = tf.argmax(predictions[0][0]).numpy()
        next_word = loader.targ_tokenizer.index_word.get(predicted_id, '')

        if next_word == '<end>':
            break
        if next_word:
            result += next_word + ' '
        
        dec_input = tf.expand_dims([predicted_id], 0)

    return result.strip()

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