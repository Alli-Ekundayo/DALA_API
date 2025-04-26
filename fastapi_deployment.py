from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Initialize FastAPI app
app = FastAPI()

# Load the trained model components
encoder = tf.keras.models.load_model("encoder_model_path")
decoder = tf.keras.models.load_model("decoder_model_path")

with open("inp_lang_tokenizer.pkl", "rb") as f:
    inp_lang_tokenizer = pickle.load(f)

with open("targ_lang_tokenizer.pkl", "rb") as f:
    targ_lang_tokenizer = pickle.load(f)

# Define input and output schemas
class TranslationRequest(BaseModel):
    itsekiri_sentence: str

class TranslationResponse(BaseModel):
    english_translation: str

# Helper function to evaluate a sentence
def evaluate_sentence(sentence):
    sentence = inp_lang_tokenizer.texts_to_sequences([sentence])
    sentence = pad_sequences(sentence, maxlen=20, padding='post')

    result = ''
    hidden = [tf.zeros((1, 512))]
    enc_out, enc_hidden = encoder(tf.convert_to_tensor(sentence), hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang_tokenizer.word_index['<start>']], 0)

    for _ in range(20):  # Max output length
        predictions, dec_hidden = decoder(dec_input, dec_hidden)
        predicted_id = tf.argmax(predictions[0][0]).numpy()
        next_word = targ_lang_tokenizer.index_word.get(predicted_id, '')

        if next_word == '<end>':
            break

        result += next_word + ' '
        dec_input = tf.expand_dims([predicted_id], 0)

    return result.strip()

# Define the prediction endpoint
@app.post("/translate", response_model=TranslationResponse)
def translate(request: TranslationRequest):
    try:
        translation = evaluate_sentence(request.itsekiri_sentence)
        return TranslationResponse(english_translation=translation)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))