import os
import logging

# --- Логирование ---
LOG_LEVEL = logging.INFO  # или DEBUG для отладки

# --- Пути к файлам ---
MODEL_PATH = "model.pkl"
ACCURACY_PATH = "accuracy.json"

# --- Настройки данных ---
HORIZON_PERIODS = 192              # 96 * 15 минут = 1 день
LOOKBACK_PERIOD = "60d"           # Исторический горизонт (не используется напрямую)
MIN_DATA_ROWS = 200

# --- Метрики модели ---
TARGET_ACCURACY = 0.9
MIN_ACCURACY_FOR_SIGNAL = 0.7

# --- Время обучения ---
MAX_TRAINING_TIME = 1200          # 20 минут
N_SPLITS_TS_CV = 7

# --- Optuna ---
OPTUNA_STORAGE_URL = "sqlite:///db.sqlite3"
OPTUNA_STUDY_NAME = "lgbm_eurusd"

# --- Порог вероятности сигнала ---
PREDICTION_PROB_THRESHOLD = 0.8

# --- Торговая логика ---
MIN_ATR_SL_MULTIPLIER = 1.5
RISK_REWARD_RATIO = 3.0
BB_BUFFER_FACTOR = 0.0001
MAX_REASONABLE_ATR = 0.005
MAX_TP_ATR_MULTIPLIER = 5.0

# --- Сервер FastAPI (Render / локально) ---
UVICORN_HOST = "0.0.0.0"
UVICORN_PORT = int(os.environ.get("PORT", 10000))
