"""
Text preprocessing for Itsekiri-English translation
"""
import re
import string
import unicodedata
import pandas as pd
import logging
from typing import Optional
from nltk.tokenize import word_tokenize
import wordninja

# Configure logging
logger = logging.getLogger(__name__)

# Import NLTK components on startup if needed
try:
    import nltk
    nltk.download('punkt', quiet=True)
except ImportError:
    print("Warning: NLTK not available - some preprocessing features may be limited.")
    
def normalize_diacritics(text: str) -> str:
    """Normalize diacritical marks in text
    
    Args:
        text (str): Input text with diacritics
        
    Returns:
        str: Text with normalized diacritics
    """
    # Keep diacritics as is - they are meaningful in Itsekiri
    return text

def clean_itsekiri_punctuation(text):
    """Cleans punctuation from Itsekiri text
    
    Args:
        text (str): Itsekiri text to clean
        
    Returns:
        str: Cleaned text
    """
    text = re.sub(r"'|'", "", text)  # Clean quotation marks
    text = re.sub(r"\/.*", "", text)  # Clean every character including and following a slash
    text = re.sub(r",|\.", " ", text)  # Replace with space
    return text

def prepare_itsekiri_input(text: str) -> str:
    """Prepares Itsekiri text for translation
    
    Args:
        text (str): Input Itsekiri text
        
    Returns:
        str: Cleaned Itsekiri text ready for translation
    """
    try:
        # Normalize text and convert to lowercase
        text = normalize_diacritics(text.strip())
        logger.debug(f"After normalization: '{text}'")
        
        text = text.lower()
        logger.debug(f"After lowercase: '{text}'")
        
        # Clean punctuation but preserve word boundaries, diacritics and hyphens
        text = re.sub(r"[^\w\s\-ẹọàáèéìíòóùúāēīōūṣ]", " ", text)
        logger.debug(f"After punctuation cleaning: '{text}'")
        
        # Normalize spaces while preserving word boundaries
        text = ' '.join(part.strip() for part in text.split())
        logger.debug(f"After space normalization: '{text}'")
        
        return text.strip()
        
    except Exception as e:
        logger.error(f"Error preprocessing Itsekiri text: {str(e)}", exc_info=True)
        return text

def fix_concatenated_words(tokens, english_wordlist=None):
    """Splits wrongly concatenated English words
    
    Args:
        tokens (list): List of word tokens
        english_wordlist (set, optional): Set of known English words
        
    Returns:
        list: List with concatenated words split
    """
    # If no wordlist provided, rely only on wordninja
    if english_wordlist is None:
        return sum([wordninja.split(token) if len(token) > 10 else [token] for token in tokens], [])
    
    new_tokens = []
    
    for token in tokens:
        if token in english_wordlist:
            new_tokens.append(token)
        else:
            subwords = wordninja.split(token)
            # Check if all subwords are English words
            if all(word in english_wordlist for word in subwords):
                new_tokens.extend(subwords)
            else:
                new_tokens.append(token)
    
    return new_tokens

def post_process_translation(text: str) -> str:
    """Post-process translated text
    
    Args:
        text (str): Raw translated text
        
    Returns:
        str: Cleaned and formatted translation
    """
    if not text:
        return ""
        
    # Capitalize first letter of sentences
    text = '. '.join(s.capitalize() for s in text.split('. '))
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text.strip()

def preprocess_text_pairs(english_texts, itsekiri_texts):
    """Preprocesses pairs of English and Itsekiri texts for training
    
    Args:
        english_texts (list): List of English texts
        itsekiri_texts (list): List of Itsekiri texts
        
    Returns:
        tuple: (preprocessed_english_texts, preprocessed_itsekiri_texts)
    """
    processed_english = []
    processed_itsekiri = []
    
    for eng, its in zip(english_texts, itsekiri_texts):
        # Skip invalid entries (NaN, None, or non-string types)
        if pd.isna(eng) or pd.isna(its) or not isinstance(eng, str) or not isinstance(its, str):
            continue
            
        # Process English text
        eng = eng.lower().strip()
        eng = re.sub(r'[^\w\s]', ' ', eng)  # Replace punctuation with space
        eng = ' '.join(eng.split())  # Normalize whitespace
        
        # Process Itsekiri text
        its = prepare_itsekiri_input(its)
        
        # Only add if both texts are non-empty after processing
        if eng and its:
            processed_english.append(eng)
            processed_itsekiri.append(its)
    
    return processed_english, processed_itsekiri