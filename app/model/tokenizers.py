"""
Tokenization utilities for the Itsekiri-English translation model.
"""
import re
import json
import string
import unicodedata
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def normalize_diacritics(text):
    """Ensures all glyphs and diacritics are consistently encoded
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text with normalized diacritics
    """
    return unicodedata.normalize("NFC", text)

def clean_itsekiri_text(text):
    """Cleans Itsekiri text by normalizing diacritics and removing unwanted characters
    
    Args:
        text (str): Input Itsekiri text
        
    Returns:
        str: Cleaned Itsekiri text
    """
    text = normalize_diacritics(text)
    text = text.lower()
    # Clean punctuation from Itsekiri word fields
    text = re.sub(r"'|'", "", text)  # Clean quotation marks
    text = re.sub(r"\/.*", "", text)  # Clean every character including and following a slash
    text = re.sub(r",|\.", " ", text)  # Replace with space
    return text.strip()

def clean_english_text(text):
    """Cleans English text
    
    Args:
        text (str): Input English text
        
    Returns:
        str: Cleaned English text with start and end tokens
    """
    text = text.lower()
    # Remove dots and unnecessary punctuation but keep important ones
    text = re.sub(r"[.]", " ", text)
    # Clean "also see" and similar references
    text = re.sub(r"also.*", "", text)
    return "<start> " + text.strip() + " <end>"

def create_tokenizers(texts, vocab_size=None, oov_token="<unk>"):
    """Creates a tokenizer fitted on the provided texts
    
    Args:
        texts (list): List of texts to fit the tokenizer on
        vocab_size (int, optional): Maximum vocabulary size
        oov_token (str, optional): Token to use for out-of-vocabulary words
        
    Returns:
        Tokenizer: Fitted tokenizer
    """
    tokenizer = Tokenizer(num_words=vocab_size, filters='', lower=True, oov_token=oov_token)
    tokenizer.fit_on_texts(texts)
    return tokenizer

def prepare_dataset(itsekiri_texts, english_texts):
    """Prepares the dataset for training by creating tokenizers and sequences
    
    Args:
        itsekiri_texts (list): List of Itsekiri texts
        english_texts (list): List of English texts
        
    Returns:
        tuple: (input_tensor, target_tensor, inp_tokenizer, targ_tokenizer,
                max_input_length, max_target_length)
    """
    # Clean texts
    cleaned_itsekiri = [clean_itsekiri_text(text) for text in itsekiri_texts]
    cleaned_english = [clean_english_text(text) for text in english_texts]
    
    # Create tokenizers
    inp_tokenizer = create_tokenizers(cleaned_itsekiri)
    targ_tokenizer = create_tokenizers(cleaned_english)
    
    # Convert texts to sequences
    input_sequences = inp_tokenizer.texts_to_sequences(cleaned_itsekiri)
    target_sequences = targ_tokenizer.texts_to_sequences(cleaned_english)
    
    # Pad sequences
    max_input_length = max(len(seq) for seq in input_sequences)
    max_target_length = max(len(seq) for seq in target_sequences)
    
    input_tensor = pad_sequences(input_sequences, maxlen=max_input_length, padding='post')
    target_tensor = pad_sequences(target_sequences, maxlen=max_target_length, padding='post')
    
    return (
        input_tensor, 
        target_tensor, 
        inp_tokenizer, 
        targ_tokenizer,
        max_input_length,
        max_target_length
    )

def save_tokenizer(tokenizer, path):
    """Saves a tokenizer to a file
    
    Args:
        tokenizer: The tokenizer to save
        path (str): Path to save the tokenizer
    """
    tokenizer_json = tokenizer.to_json()
    with open(path, 'w', encoding='utf-8') as f:
        f.write(tokenizer_json)
        
def load_tokenizer(path):
    """Loads a tokenizer from a file
    
    Args:
        path (str): Path to the saved tokenizer
        
    Returns:
        Tokenizer: Loaded tokenizer
    """
    with open(path, 'r', encoding='utf-8') as f:
        tokenizer_json = f.read()
    
    from tensorflow.keras.preprocessing.text import tokenizer_from_json
    return tokenizer_from_json(tokenizer_json)

def create_or_load_tokenizers(itsekiri_texts, english_texts, itsekiri_tokenizer_path=None, english_tokenizer_path=None):
    """Creates new tokenizers or loads existing ones if available
    
    Args:
        itsekiri_texts (list): List of Itsekiri texts
        english_texts (list): List of English texts
        itsekiri_tokenizer_path (str, optional): Path to saved Itsekiri tokenizer
        english_tokenizer_path (str, optional): Path to saved English tokenizer
        
    Returns:
        tuple: (itsekiri_tokenizer, english_tokenizer)
    """
    try:
        if itsekiri_tokenizer_path and english_tokenizer_path:
            # Try to load existing tokenizers
            itsekiri_tokenizer = load_tokenizer(itsekiri_tokenizer_path)
            english_tokenizer = load_tokenizer(english_tokenizer_path)
            return itsekiri_tokenizer, english_tokenizer
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    
    # Create new tokenizers if loading failed or paths weren't provided
    cleaned_itsekiri = [clean_itsekiri_text(text) for text in itsekiri_texts]
    cleaned_english = [clean_english_text(text) for text in english_texts]
    
    itsekiri_tokenizer = create_tokenizers(cleaned_itsekiri)
    english_tokenizer = create_tokenizers(cleaned_english)
    
    # Save the new tokenizers if paths were provided
    if itsekiri_tokenizer_path:
        save_tokenizer(itsekiri_tokenizer, itsekiri_tokenizer_path)
    if english_tokenizer_path:
        save_tokenizer(english_tokenizer, english_tokenizer_path)
    
    return itsekiri_tokenizer, english_tokenizer