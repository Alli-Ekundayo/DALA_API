"""
Neural machine translation model for Itsekiri-English translation.
Implements an encoder-decoder architecture with GRU units.
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
import json
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class Encoder(tf.keras.Model):
    """Encoder model for the seq2seq translation system"""
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super().__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(enc_units,
                                      return_sequences=True,
                                      return_state=True,
                                      recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_sz
        return tf.zeros((batch_size, self.enc_units))


class Decoder(tf.keras.Model):
    """Decoder model for the seq2seq translation system"""
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super().__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(dec_units,
                                      return_sequences=True,
                                      return_state=True,
                                      recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        x = self.fc(output)
        return x, state


class Translator:
    """Main translator class that handles the end-to-end translation process"""
    def __init__(self, config):
        """Initialize the translator model
        
        Args:
            config: Dictionary containing model configuration
        """
        self.config = config
        self.embedding_dim = config.get('embedding_dim', 256)
        self.units = config.get('units', 512)
        self.batch_size = config.get('batch_size', 1)  # Default to 1 for inference
        
        # Initialize tokenizers
        self.inp_lang_tokenizer = None
        self.targ_lang_tokenizer = None
        
        # Initialize model
        self.encoder = None
        self.decoder = None
        
        # Model metadata
        self.input_vocab_size = 0
        self.target_vocab_size = 0
        self.max_input_length = 0
        self.max_target_length = 0
        
        # Add dictionary for direct lookups
        self.dictionary = {}
        
    def build_model(self, input_vocab_size, target_vocab_size):
        """Build the encoder and decoder models
        
        Args:
            input_vocab_size: Size of the input vocabulary
            target_vocab_size: Size of the target vocabulary
        """
        self.input_vocab_size = input_vocab_size
        self.target_vocab_size = target_vocab_size
        
        self.encoder = Encoder(
            input_vocab_size, 
            self.embedding_dim, 
            self.units, 
            self.batch_size
        )
        
        self.decoder = Decoder(
            target_vocab_size, 
            self.embedding_dim, 
            self.units, 
            self.batch_size
        )
    
    def load_tokenizers(self, inp_tokenizer_path, targ_tokenizer_path):
        """Load tokenizers from saved files
        
        Args:
            inp_tokenizer_path: Path to the input language tokenizer
            targ_tokenizer_path: Path to the target language tokenizer
        """
        with open(inp_tokenizer_path, 'r', encoding='utf-8') as f:
            inp_tokenizer_data = json.load(f)
            
        with open(targ_tokenizer_path, 'r', encoding='utf-8') as f:
            targ_tokenizer_data = json.load(f)
            
        from tensorflow.keras.preprocessing.text import tokenizer_from_json
        self.inp_lang_tokenizer = tokenizer_from_json(json.dumps(inp_tokenizer_data))
        self.targ_lang_tokenizer = tokenizer_from_json(json.dumps(targ_tokenizer_data))
        
        # Update vocabulary sizes
        self.input_vocab_size = len(self.inp_lang_tokenizer.word_index) + 1
        self.target_vocab_size = len(self.targ_lang_tokenizer.word_index) + 1
        
    def save_model(self, model_dir):
        """Save the model weights and tokenizers
        
        Args:
            model_dir: Directory to save the model
        """
        os.makedirs(model_dir, exist_ok=True)
        
        # Save encoder and decoder weights with correct file extension
        self.encoder.save_weights(os.path.join(model_dir, 'encoder.weights.h5'))
        self.decoder.save_weights(os.path.join(model_dir, 'decoder.weights.h5'))
        
        # Save model metadata
        metadata = {
            'input_vocab_size': self.input_vocab_size,
            'target_vocab_size': self.target_vocab_size,
            'embedding_dim': self.embedding_dim,
            'units': self.units,
            'max_input_length': self.max_input_length,
            'max_target_length': self.max_target_length
        }
        
        with open(os.path.join(model_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)
            
        # Save tokenizers
        with open(os.path.join(model_dir, 'inp_tokenizer.json'), 'w', encoding='utf-8') as f:
            f.write(self.inp_lang_tokenizer.to_json())
            
        with open(os.path.join(model_dir, 'targ_tokenizer.json'), 'w', encoding='utf-8') as f:
            f.write(self.targ_lang_tokenizer.to_json())
    
    def load_model(self, model_dir):
        """Load the model from saved weights
        
        Args:
            model_dir: Directory containing the saved model
        """
        # Load metadata
        with open(os.path.join(model_dir, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
            
        self.input_vocab_size = metadata['input_vocab_size']
        self.target_vocab_size = metadata['target_vocab_size']
        self.embedding_dim = metadata['embedding_dim']
        self.units = metadata['units']
        self.max_input_length = metadata['max_input_length']
        self.max_target_length = metadata['max_target_length']
        
        # Load tokenizers
        self.load_tokenizers(
            os.path.join(model_dir, 'inp_tokenizer.json'),
            os.path.join(model_dir, 'targ_tokenizer.json')
        )
        
        # Build model
        self.build_model(self.input_vocab_size, self.target_vocab_size)
        
        # Create sample inputs to build the model
        dummy_input = tf.zeros((1, 1))
        dummy_hidden = self.encoder.initialize_hidden_state(batch_size=1)
        
        # Build the models by calling them
        self.encoder(dummy_input, dummy_hidden)
        decoder_input = tf.zeros((1, 1))
        self.decoder(decoder_input, dummy_hidden)
        
        # Load weights with correct file extension
        self.encoder.load_weights(os.path.join(model_dir, 'encoder.weights.h5'))
        self.decoder.load_weights(os.path.join(model_dir, 'decoder.weights.h5'))
    
    def load_dictionary(self, data_path='data/clean_itsekiri.csv'):
        """Load the dictionary from CSV file"""
        try:
            logger.info(f"Loading dictionary from {data_path}")
            df = pd.read_csv(data_path)
            # Create dictionary mapping from Itsekiri words to English translations
            self.dictionary = {}
            for _, row in df.iterrows():
                itsekiri_word = str(row['new_target']).lower().strip()
                english_trans = str(row['clean_translation']).strip()
                if itsekiri_word and english_trans and itsekiri_word != 'nan' and english_trans != 'nan':
                    if itsekiri_word in self.dictionary:
                        # If word exists, append new meaning
                        if english_trans not in self.dictionary[itsekiri_word]:
                            self.dictionary[itsekiri_word] = f"{self.dictionary[itsekiri_word]} / {english_trans}"
                    else:
                        self.dictionary[itsekiri_word] = english_trans
            logger.info(f"Loaded {len(self.dictionary)} dictionary entries")
            # Log a few sample entries for verification
            sample_entries = list(self.dictionary.items())[:5]
            logger.debug(f"Sample dictionary entries: {sample_entries}")
        except Exception as e:
            logger.error(f"Failed to load dictionary: {str(e)}", exc_info=True)
    
    def translate(self, sentence, max_length=20):
        """Translate an Itsekiri sentence to English"""
        try:
            # Clean and preprocess input
            cleaned = sentence.lower().strip()
            
            # Try direct dictionary lookup first
            if cleaned in self.dictionary:
                return self.dictionary[cleaned]
                
            # If compound word, try translating parts
            if ' ' in cleaned or '-' in cleaned:
                parts = cleaned.replace('-', ' ').split()
                translations = []
                for part in parts:
                    if part in self.dictionary:
                        translations.append(self.dictionary[part])
                if translations:
                    return ' + '.join(translations)
            
            # If no direct match, try neural translation
            translated = self._neural_translate(sentence, max_length)
            if translated.strip():
                return translated
                
            return ""  # Return empty string if no translation found
            
        except Exception as e:
            logger.error(f"Translation error: {str(e)}", exc_info=True)
            return ""
        
    def _neural_translate(self, sentence, max_length=20):
        """Internal method for neural translation"""
        # Tokenize and pad the input sentence
        sequence = self.inp_lang_tokenizer.texts_to_sequences([sentence])
        padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(
            sequence, 
            maxlen=self.max_input_length if self.max_input_length > 0 else 50,
            padding='post'
        )
        
        # Initialize results
        result = []
        
        # Initialize encoder state
        enc_hidden = self.encoder.initialize_hidden_state(batch_size=1)
        enc_output, enc_hidden = self.encoder(tf.convert_to_tensor(padded_sequence), enc_hidden)
        
        # Initialize decoder input and state
        dec_hidden = enc_hidden
        
        # Get the <start> token ID
        start_token_id = self.targ_lang_tokenizer.word_index.get('<start>', 1)
        dec_input = tf.expand_dims([start_token_id], 0)
        
        end_token_id = self.targ_lang_tokenizer.word_index.get('<end>', 2)
        
        for t in range(max_length):
            predictions, dec_hidden = self.decoder(dec_input, dec_hidden)
            
            # Get the predicted ID
            predicted_id = tf.argmax(predictions[0][0]).numpy()
            
            # Check if it's the end token
            if predicted_id == end_token_id:
                break
            
            # Get the word from the ID
            word = self.targ_lang_tokenizer.index_word.get(predicted_id, '')
            if word and word not in ['<start>', '<end>', '<unk>']:
                result.append(word)
            
            # Update decoder input
            dec_input = tf.expand_dims([predicted_id], 0)
        
        return ' '.join(result)