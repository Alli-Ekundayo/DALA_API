"""
Main application module for the Itsekiri-English translation API.
"""
import os
import logging
from flask import Flask
from app.config import get_config
from app.model.translator import Translator
from app.api import api_bp

def setup_logging(app):
    """Configure logging for the application
    
    Args:
        app (Flask): Flask application
    """
    log_level = getattr(logging, app.config['LOG_LEVEL'])
    
    # Configure Flask logger
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    app.logger.addHandler(handler)
    app.logger.setLevel(log_level)
    
    # Set level for other loggers
    logging.getLogger('werkzeug').setLevel(log_level)

def create_app(config_name=None):
    """Create and configure the Flask application
    
    Args:
        config_name (str, optional): Configuration name to use
        
    Returns:
        Flask: Configured Flask application
    """
    # Create Flask app
    app = Flask(__name__)
    
    # Load configuration
    if config_name is None:
        config_name = os.getenv('FLASK_ENV', 'development')
    
    config = get_config(config_name)
    app.config.from_object(config)
    
    # Setup logging
    setup_logging(app)
    
    # Initialize the translator model
    translator = Translator()
    app.translator = translator
    
    # Register blueprints
    app.register_blueprint(api_bp, url_prefix='/api')
    
    @app.route('/health')
    def health_check():
        """Basic health check endpoint"""
        return {'status': 'healthy'}, 200
    
    return app