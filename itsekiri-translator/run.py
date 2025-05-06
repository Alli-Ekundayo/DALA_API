#!/usr/bin/env python
# Entry point for the API

from app import api
from app.config import DevelopmentConfig, ProductionConfig
import os

if __name__ == '__main__':
    # Set configuration based on environment
    config = ProductionConfig if os.environ.get('ENVIRONMENT') == 'production' else DevelopmentConfig
    
    # Run the Flask application
    api.create_app(config).run(
        host=config.HOST,
        port=config.PORT,
        debug=config.DEBUG
    )