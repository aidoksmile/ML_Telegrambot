import logging
import os

# --- General Settings ---
LOG_LEVEL = logging.INFO # Уровень логирования (DEBUG, INFO, WARNING, ERROR, CRITICAL)

# --- Model Paths ---
MODEL_PATH = "model.pkl"
ACCURACY_PATH = "accuracy.json"

# --- Strategy Parameters ---
HORIZON_PERIODS = 1
LOOKBACK_PERIOD = "60d" # <-- ИЗМЕНЕНО: Максимум для 15-минутных данных
MIN_DATA_ROWS = 100 # Возможно, потребуется увеличить для 15-минутных данных, так как их будет больше
TARGET_ACCURACY = 0.8
MIN_ACCURACY_FOR_SIGNAL = 0.5
MAX_TRAINING_TIME = 3600 # Максимальное время обучения в секундах (1 час)
PREDICTION_PROB_THRESHOLD = 0.55 # Порог вероятности для сигнала BUY/SELL. Если ниже, то HOLD.

# --- Time Series Split Parameters ---
N_SPLITS_TS_CV = 3 # Количество разбиений для TimeSeriesSplit в Optuna

# --- Optuna Settings ---
OPTUNA_STORAGE_URL = "sqlite:///optuna_study.db"
OPTUNA_STUDY_NAME = "lgbm_eurusd_study"

# --- FastAPI Server Settings ---
UVICORN_HOST = "0.0.0.0"
UVICORN_PORT = int(os.environ.get("PORT", 10000))
