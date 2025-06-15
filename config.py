import logging
import os

# --- General Settings ---
LOG_LEVEL = logging.INFO # Уровень логирования (DEBUG, INFO, WARNING, ERROR, CRITICAL)

# --- Model Paths ---
MODEL_PATH = "model.h5" # Изменено на .h5 для Keras моделей
ACCURACY_PATH = "accuracy.json"

# --- Strategy Parameters ---
HORIZON_PERIODS = 96 # 1 день (96 * 15-минутных интервалов)
LOOKBACK_PERIOD = "60d" # Максимум для 15-минутных данных от Yahoo Finance
MIN_DATA_ROWS = 500 # Увеличено для создания последовательностей и горизонта предсказания
TARGET_ACCURACY = 0.8 # Целевая точность (F1-score), может быть сложнее достичь с NN
MIN_ACCURACY_FOR_SIGNAL = 0.5
MAX_TRAINING_TIME = 7200 # Увеличено до 2 часов для обучения нейронной сети
PREDICTION_PROB_THRESHOLD = 0.55 # Порог вероятности для сигнала BUY/SELL. Если ниже, то HOLD.

# --- Time Series Split Parameters ---
N_SPLITS_TS_CV = 3 # Количество разбиений для TimeSeriesSplit в Optuna

# --- Optuna Settings ---
OPTUNA_STORAGE_URL = "sqlite:///optuna_study_nn.db" # Новая база данных для исследования NN
OPTUNA_STUDY_NAME = "nn_eurusd_study" # Новое имя исследования

# --- Neural Network Settings ---
SEQUENCE_LENGTH = 40 # Количество предыдущих 15-минутных баров для использования в качестве входной последовательности
NN_EPOCHS = 100 # Максимальное количество эпох, будет использоваться EarlyStopping
NN_BATCH_SIZE = 32 # Размер батча для обучения NN
NN_PATIENCE = 10 # Количество эпох без улучшения для Early Stopping

# --- FastAPI Server Settings ---
UVICORN_HOST = "0.0.0.0"
UVICORN_PORT = int(os.environ.get("PORT", 10000))
