import os
import logging

# --- Logging ---
LOG_LEVEL = logging.INFO  # Можно установить logging.DEBUG для подробного вывода

# --- Модель ---
MODEL_PATH = "model.pkl"                  # Путь к файлу с моделью
ACCURACY_PATH = "accuracy.json"          # Путь к файлу с метриками

# --- Данные и признаки ---
HORIZON_PERIODS = 96                     # 96 x 15m = 1 день вперёд
LOOKBACK_PERIOD = "60d"                 # Только для истории — не используется напрямую (заменено на days=60)
MIN_DATA_ROWS = 100                     # Минимум строк для обучения

# --- Цели по метрикам ---
TARGET_ACCURACY = 0.6                   # Целевая точность (F1-score)
MIN_ACCURACY_FOR_SIGNAL = 0.5          # Минимум, при котором сигнал разрешён

# --- Обучение ---
MAX_TRAINING_TIME = 1800               # В секундах (30 минут)
N_SPLITS_TS_CV = 5                     # Кросс-валидация по времени

# --- Оптимизация Optuna ---
OPTUNA_STORAGE_URL = "sqlite:///db.sqlite3"
OPTUNA_STUDY_NAME = "lgbm_eurusd_study"

# --- Стратегия сигналов ---
PREDICTION_PROB_THRESHOLD = 0.55       # Если выше — BUY, ниже — SELL
MIN_ATR_SL_MULTIPLIER = 1.5            # Минимальный SL = ATR * множитель
RISK_REWARD_RATIO = 2.0                # Соотношение TP к риску
BB_BUFFER_FACTOR = 0.0001              # Маленький отступ от Bollinger Band

# --- Ограничения ATR и TP ---
MAX_REASONABLE_ATR = 0.001             # Ограничение на слишком большой ATR
MAX_TP_ATR_MULTIPLIER = 5.0            # TP не дальше чем 5 * ATR

# --- FastAPI / Uvicorn ---
UVICORN_HOST = "0.0.0.0"
UVICORN_PORT = int(os.environ.get("PORT", 10000))
