import logging

# --- General Settings ---
LOG_LEVEL = logging.INFO # Уровень логирования (DEBUG, INFO, WARNING, ERROR, CRITICAL)

# --- Model Paths ---
MODEL_PATH = "model.pkl"
ACCURACY_PATH = "accuracy.json"

# --- Strategy Parameters ---
HORIZON_PERIODS = 1
LOOKBACK_PERIOD = "5y" # Изменено с "max" на 5 лет для оптимизации загрузки данных
MIN_DATA_ROWS = 100
TARGET_ACCURACY = 0.8
MIN_ACCURACY_FOR_SIGNAL = 0.5
MAX_TRAINING_TIME = 3600 # Максимальное время обучения в секундах (1 час)
PREDICTION_PROB_THRESHOLD = 0.55 # Порог вероятности для сигнала BUY/SELL. Если ниже, то HOLD.

# --- Time Series Split Parameters ---
N_SPLITS_TS_CV = 3 # Количество разбиений для TimeSeriesSplit в Optuna

# --- FastAPI Server Settings ---
UVICORN_HOST = "0.0.0.0"
UVICORN_PORT = int(os.environ.get("PORT", 10000)) # Используем переменную окружения PORT, если есть
