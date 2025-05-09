"""
Configuration settings for the translation API.
"""
import os

class Config:
    """Base configuration"""
    # App settings
    DEBUG = False
    TESTING = False
    VERSION = "1.0.0"
    HOST = os.environ.get('HOST', '0.0.0.0')  # Default host that allows external connections
    PORT = int(os.environ.get('PORT', 5000))  # Default Flask port
    
    # Model settings
    MODEL_DIR = os.environ.get('MODEL_DIR', 'models')
    MODEL_PATH = os.environ.get('MODEL_PATH', 'models/itsekiri_translator_20250507_215757')  # Latest trained model
    BATCH_SIZE = 1
    EMBEDDING_DIM = 256
    UNITS = 512
    
    # API settings
    MAX_CONTENT_LENGTH = 100 * 1024  # 100KB max request size
    
    # Logging
    LOG_LEVEL = 'INFO'
    
class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
    LOG_LEVEL = 'DEBUG'

class ProductionConfig(Config):
    """Production configuration"""
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')

# Configuration mapping
config_by_name = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

def get_config():
    """Gets the current configuration based on the environment
    
    Returns:
        Config: Configuration object
    """
    env = os.environ.get('FLASK_ENV', 'default')
    return config_by_name.get(env, config_by_name['default'])