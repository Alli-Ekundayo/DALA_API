"""
API endpoints for the Itsekiri-English translation service.
"""
from flask import Flask, Blueprint, request, jsonify, current_app
import time
import logging
from app.preprocessing import prepare_itsekiri_input, post_process_translation
from app.model.translator import Translator
import os

# Create a Flask Blueprint for API routes
api_bp = Blueprint('api', __name__)

# Configure logging
logger = logging.getLogger(__name__)

@api_bp.route('/', methods=['GET'])
def index():
    """Root endpoint that provides API information"""
    return jsonify({
        'name': 'Itsekiri-English Translator API',
        'version': current_app.config.get('VERSION', '1.0.0'),
        'status': 'healthy' if current_app.translator is not None else 'model not loaded',
        'endpoints': {
            '/': 'Get API information (this endpoint)',
            '/api/translate': 'POST - Translate Itsekiri text to English'
        }
    })

def create_app(config):
    """Create and configure the Flask application
    
    Args:
        config: Configuration object with app settings
        
    Returns:
        Flask application instance
    """
    app = Flask(__name__)
    app.config.from_object(config)
    
    # Add root route that redirects to API info
    @app.route('/')
    def root():
        return jsonify({
            'name': 'Itsekiri-English Translator API',
            'version': app.config.get('VERSION', '1.0.0'),
            'status': 'healthy' if app.translator is not None else 'model not loaded',
            'docs': 'Visit /api for full API documentation and endpoints'
        })
    
    # Register blueprint
    app.register_blueprint(api_bp, url_prefix='/api')
    
    # Load the translator model
    with app.app_context():
        model_path = app.config.get('MODEL_PATH', 'models/latest')
        try:
            # Ensure model path exists
            if not os.path.exists(model_path):
                raise ValueError(f"Model path does not exist: {model_path}")
            
            # Load configuration for the model
            model_config = {
                'embedding_dim': app.config.get('EMBEDDING_DIM', 256),
                'units': app.config.get('UNITS', 512),
                'batch_size': app.config.get('BATCH_SIZE', 1)
            }
            
            # Initialize and load the model
            app.translator = Translator(model_config)
            app.translator.load_model(model_path)
            
            # Load the dictionary data
            dictionary_path = os.path.join('data', 'clean_itsekiri.csv')
            if os.path.exists(dictionary_path):
                app.translator.load_dictionary(dictionary_path)
            
            logger.info(f"Successfully loaded translator model from {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load translator model: {str(e)}", exc_info=True)
            app.translator = None
    
    return app

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
    """Translate Itsekiri text to English"""
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
        logger.info(f"Preprocessed text: '{preprocessed_text}'")
        
        # Translate
        model = current_app.translator
        if model is None:
            raise ValueError("Translator model is not loaded")
            
        # Tokenize and check if tokens are in vocabulary
        sequence = model.inp_lang_tokenizer.texts_to_sequences([preprocessed_text])
        logger.info(f"Input sequence: {sequence}")
        
        translated_text = model.translate(preprocessed_text)
        logger.info(f"Raw translation: '{translated_text}'")
        
        # Post-process result
        final_text = post_process_translation(translated_text)
        logger.info(f"Final translation: '{final_text}'")
        
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