import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    DEBUG = False
    TESTING = False
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-key-change-in-production")

    MODEL_PATH = os.getenv("MODEL_PATH", "models/fraud_model_xgboost.pkl")
    SCALER_PATH = os.getenv("SCALER_PATH", "models/feature_scaler.pkl")
    FEATURES_PATH = os.getenv("FEATURES_PATH", "models/feature_names.pkl")

    MAX_CONTENT_LENGTH = 16 * 1024 * 1024

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False

class TestingConfig(Config):
    TESTING = True
    DEBUG = True

config = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
    "default": DevelopmentConfig,
}
