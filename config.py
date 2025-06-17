import os
import logging

# --- General Settings ---
LOG_LEVEL = logging.INFO # Уровень логирования (DEBUG, INFO, WARNING, ERROR, CRITICAL)

# --- Model Paths ---
MODEL_PATH = "model.pkl" # Путь для сохранения обученной модели
ACCURACY_PATH = "accuracy.json" # Путь для сохранения метрик точности

# --- Data & Feature Engineering Parameters ---
HORIZON_PERIODS = 96 # Горизонт предсказания (на сколько периодов вперед предсказываем). 1 день = 96 15-минутных периодов.
LOOKBACK_PERIOD = "60d" # Период загрузки исторических данных (например, "60d" для 60 дней, "1y" для 1 года)
MIN_DATA_ROWS = 100 # Минимальное количество строк данных для обучения модели

# --- Training & Optimization Parameters ---
TARGET_ACCURACY = 0.6 # Целевая точность/F1-score для модели (если ниже, модель переобучается)
MIN_ACCURACY_FOR_SIGNAL = 0.5 # Минимальная точность/F1-score для генерации торговых сигналов
MAX_TRAINING_TIME = 3600 # Максимальное время обучения модели в секундах (1 час)
N_SPLITS_TS_CV = 3 # Количество фолдов для TimeSeriesSplit кросс-валидации в Optuna

# --- Optuna Settings ---
OPTUNA_STORAGE_URL = "sqlite:///db.sqlite3" # URL для хранения результатов Optuna (можно использовать PostgreSQL, MySQL и т.д.)
OPTUNA_STUDY_NAME = "lgbm_eurusd_study" # Имя исследования Optuna

# --- Signal Generation Parameters ---
PREDICTION_PROB_THRESHOLD = 0.55 # Порог вероятности для генерации сигнала (например, >0.55 для BUY, <0.45 для SELL)

# --- Data & Feature Engineering Parameters ---
# ... (остальные параметры)
TARGET_PRICE_CHANGE_THRESHOLD = 0.0005 # Порог изменения цены для целевой переменной (например, 0.05%)


# --- FastAPI/Uvicorn Settings ---
UVICORN_HOST = "0.0.0.0"
UVICORN_PORT = int(os.environ.get("PORT", 10000)) # Используем переменную окружения PORT, если есть
