"""
Text preprocessing utilities for the translation API.
"""
import re
import string
import unicodedata
from nltk.tokenize import word_tokenize
import wordninja

# Import NLTK components on startup if needed
try:
    import nltk
    nltk.download('punkt', quiet=True)
except ImportError:
    print("Warning: NLTK not available - some preprocessing features may be limited.")
    
def normalize_diacritics(text):
    """Normalizes diacritics in text to ensure consistent encoding
    
    Args:
        text (str): Text to normalize
        
    Returns:
        str: Text with normalized diacritics
    """
    return unicodedata.normalize("NFC", text)

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

def prepare_itsekiri_input(text):
    """Prepares Itsekiri text for translation
    
    Args:
        text (str): Input Itsekiri text
        
    Returns:
        str: Cleaned Itsekiri text ready for translation
    """
    # Normalize and lowercase
    text = normalize_diacritics(text)
    text = text.lower()
    
    # Clean punctuation
    text = clean_itsekiri_punctuation(text)
    
    return text.strip()

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

def post_process_translation(text):
    """Post-processes a translated English text
    
    Args:
        text (str): Raw translated text
        
    Returns:
        str: Post-processed text
    """
    # Capitalize first letter
    if text:
        text = text[0].upper() + text[1:]
    
    # Fix spacing around punctuation
    text = re.sub(r'\s+([,.?!:;])', r'\1', text)
    
    # Fix spacing after punctuation
    text = re.sub(r'([,.?!:;])([^\s])', r'\1 \2', text)
    
    return text