import os
import logging

# --- Логирование ---
LOG_LEVEL = logging.INFO  # INFO или DEBUG

# --- Пути к файлам ---
MODEL_PATH = "model.pkl"
ACCURACY_PATH = "accuracy.json"

# --- Настройки данных ---
HORIZON_PERIODS = 100              # Сдвиг таргета: ~1 день на 15-минутках
LOOKBACK_PERIOD = "60d"           # Исторический горизонт (опционально)
MIN_DATA_ROWS = 200

# --- Метрики модели ---
TARGET_ACCURACY = 0.9
MIN_ACCURACY_FOR_SIGNAL = 0.7

# --- Время обучения ---
MAX_TRAINING_TIME = 7200           # 2 часа
N_SPLITS_TS_CV = 7                # Кросс-валидация

# --- Optuna ---
OPTUNA_STORAGE_URL = "sqlite:///db.sqlite3"
OPTUNA_STUDY_NAME = "lgbm_eurusd"

# --- Порог вероятности сигнала ---
PREDICTION_PROB_THRESHOLD = 0.8

# --- Торговая логика (использовалась ранее, но может пригодиться) ---
MIN_ATR_SL_MULTIPLIER = 1.5
RISK_REWARD_RATIO = 3.0
BB_BUFFER_FACTOR = 0.0001
MAX_REASONABLE_ATR = 0.005
MAX_TP_ATR_MULTIPLIER = 5.0

# --- Сервер FastAPI ---
UVICORN_HOST = "0.0.0.0"
UVICORN_PORT = int(os.environ.get("PORT", 10000))
