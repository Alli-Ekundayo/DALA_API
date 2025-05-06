"""
API endpoints for the Itsekiri-English translation service.
"""
from flask import Blueprint, request, jsonify, current_app
import time
import logging
from app.preprocessing import prepare_itsekiri_input, post_process_translation

# Create a Flask Blueprint for API routes
api_bp = Blueprint('api', __name__)

# Configure logging
logger = logging.getLogger(__name__)

@api_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint
    
    Returns:
        JSON response with service status
    """
    return jsonify({
        'status': 'healthy',
        'model_loaded': current_app.translator is not None,
        'version': current_app.config.get('VERSION', 'unknown')
    })

@api_bp.route('/translate', methods=['POST'])
def translate():
    """Translate Itsekiri text to English
    
    Expects a JSON request with the following format:
    {
        "text": "Itsekiri text to translate"
    }
    
    Returns:
        JSON response with the translated text
    """
    start_time = time.time()
    
    # Validate request
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400
    
    data = request.get_json()
    
    if 'text' not in data:
        return jsonify({'error': 'Request must contain a "text" field'}), 400
    
    input_text = data['text']
    if not input_text or not isinstance(input_text, str):
        return jsonify({'error': 'Text must be a non-empty string'}), 400
    
    try:
        # Preprocess input
        preprocessed_text = prepare_itsekiri_input(input_text)
        
        # Translate
        model = current_app.translator
        translated_text = model.translate(preprocessed_text)
        
        # Post-process result
        final_text = post_process_translation(translated_text)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Log the translation request (without sensitive data)
        logger.info(f"Translation processed in {processing_time:.3f}s. "
                    f"Input length: {len(input_text)}, Output length: {len(final_text)}")
        
        return jsonify({
            'input': input_text,
            'translated': final_text,
            'processing_time': processing_time
        })
        
    except Exception as e:
        logger.error(f"Translation error: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Translation failed',
            'message': str(e)
        }), 500

@api_bp.route('/batch-translate', methods=['POST'])
def batch_translate():
    """Translate multiple Itsekiri texts to English
    
    Expects a JSON request with the following format:
    {
        "texts": ["Itsekiri text 1", "Itsekiri text 2", ...]
    }
    
    Returns:
        JSON response with the translated texts
    """
    start_time = time.time()
    
    # Validate request
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400
    
    data = request.get_json()
    
    if 'texts' not in data or not isinstance(data['texts'], list):
        return jsonify({'error': 'Request must contain a "texts" array'}), 400
    
    input_texts = data['texts']
    
    try:
        results = []
        model = current_app.translator
        
        for text in input_texts:
            if not text or not isinstance(text, str):
                results.append({
                    'input': text,
                    'translated': '',
                    'error': 'Text must be a non-empty string'
                })
                continue
                
            # Preprocess input
            preprocessed_text = prepare_itsekiri_input(text)
            
            # Translate
            translated_text = model.translate(preprocessed_text)
            
            # Post-process result
            final_text = post_process_translation(translated_text)
            
            results.append({
                'input': text,
                'translated': final_text
            })
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Log the batch translation request
        logger.info(f"Batch translation processed in {processing_time:.3f}s. "
                    f"Items: {len(input_texts)}")
        
        return jsonify({
            'results': results,
            'processing_time': processing_time,
            'count': len(results)
        })
        
    except Exception as e:
        logger.error(f"Batch translation error: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Batch translation failed',
            'message': str(e)
        }), 500