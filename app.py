import yfinance as yf
import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
import json
from fastapi import FastAPI
import uvicorn
from datetime import datetime, timedelta
from send_telegram import send_telegram_message
import time
import optuna
import logging
import optuna.samplers
import config
import warnings
from threading import Thread
from optuna.exceptions import TrialPruned # Импортируем TrialPruned

# Игнорировать UserWarning из Optuna, если они связаны с повторным сообщением шагов
warnings.filterwarnings("ignore", message="The reported value is ignored because this `step` is already reported.", category=UserWarning)
# Игнорировать FutureWarning от sklearn.utils.deprecation
warnings.filterwarnings("ignore", category=FutureWarning)
# Игнорировать UserWarning от LightGBM, связанные с np.ndarray subset
warnings.filterwarnings("ignore", message="Usage of np.ndarray subset \(sliced data\) is not recommended due to it will double the peak memory cost in LightGBM.", category=UserWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


# --- Настройка логирования ---
logging.basicConfig(level=config.LOG_LEVEL, format="""%(asctime)s - %(levelname)s - %(message)s""")
logger = logging.getLogger(__name__)

app = FastAPI()

# --- Константы из config.py ---
MODEL_PATH = config.MODEL_PATH
ACCURACY_PATH = config.ACCURACY_PATH
HORIZON_PERIODS = config.HORIZON_PERIODS
LOOKBACK_PERIOD = config.LOOKBACK_PERIOD
MIN_DATA_ROWS = config.MIN_DATA_ROWS
TARGET_ACCURACY = config.TARGET_ACCURACY
MIN_ACCURACY_FOR_SIGNAL = config.MIN_ACCURACY_FOR_SIGNAL
MAX_TRAINING_TIME = config.MAX_TRAINING_TIME
PREDICTION_PROB_THRESHOLD = config.PREDICTION_PROB_THRESHOLD
N_SPLITS_TS_CV = config.N_SPLITS_TS_CV
OPTUNA_STORAGE_URL = config.OPTUNA_STORAGE_URL
OPTUNA_STUDY_NAME = config.OPTUNA_STUDY_NAME
TARGET_PRICE_CHANGE_THRESHOLD = config.TARGET_PRICE_CHANGE_THRESHOLD # Новый параметр из config

# --- Вспомогательные функции для технических индикаторов ---
def compute_rsi(data, periods=14):
    delta = data.diff()
    gain = delta.where(delta > 0, 0).rolling(window=periods).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=periods).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_bollinger_bands(data, window=20, num_std=2):
    ma = data.rolling(window=window).mean()
    std = data.rolling(window=window).std()
    upper = ma + (num_std * std)
    lower = ma - (num_std * std)
    return upper, lower

def compute_macd(data, fast=12, slow=26, signal=9):
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def compute_stochastic_oscillator(high, low, close, k_period=14, d_period=3):
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k_percent = ((close - lowest_low) / (highest_high - lowest_low + 1e-10)) * 100
    d_percent = k_percent.rolling(window=d_period).mean()
    return k_percent, d_percent

def compute_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(span=period, adjust=False).mean()
    return atr

def compute_roc(data, period=12):
    roc = ((data - data.shift(period)) / data.shift(period)) * 100
    return roc

def compute_adx(high, low, close, period=14):
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(span=period, adjust=False).mean()

    plus_di = (plus_dm.ewm(span=period, adjust=False).mean() / atr) * 100
    minus_di = (abs(minus_dm).ewm(span=period, adjust=False).mean() / atr) * 100
    
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)) * 100
    adx = dx.ewm(span=period, adjust=False).mean()
    return adx

def compute_psar(high, low, close, acceleration=0.02, maximum=0.2):
    # Инициализация
    psar = close.copy() # Начальное значение PSAR
    ep = close.copy() # Extreme Point
    af = acceleration # Acceleration Factor
    
    bull = True # Флаг бычьего тренда
    
    for i in range(1, len(close)):
        if bull:
            psar[i] = psar[i-1] + af * (ep[i-1] - psar[i-1])
            if low[i] < psar[i]: # Разворот тренда
                bull = False
                psar[i] = ep[i-1]
                af = acceleration
                ep[i] = low[i]
            else:
                if high[i] > ep[i-1]:
                    af = min(af + acceleration, maximum)
                    ep[i] = high[i]
                else:
                    ep[i] = ep[i-1]
        else: # Медвежий тренд
            psar[i] = psar[i-1] - af * (psar[i-1] - ep[i-1])
            if high[i] > psar[i]: # Разворот тренда
                bull = True
                psar[i] = ep[i-1]
                af = acceleration
                ep[i] = high[i]
            else:
                if low[i] < ep[i-1]:
                    af = min(af + acceleration, maximum)
                    ep[i] = low[i]
                else:
                    ep[i] = ep[i-1]
    return psar

# --- Пользовательская метрика F1-score для LightGBM (для LGBMClassifier) ---
def lgbm_f1_score(y_pred, y_true):
    y_true_binary = y_true.astype(int)
    y_pred_binary = (y_pred > 0.5).astype(int) # Преобразуем вероятности в бинарные предсказания
    return 'f1_score', f1_score(y_true_binary, y_pred_binary, average='weighted'), True # True означает, что чем выше, тем лучше

# --- Пользовательская метрика F1-score для LightGBM (для lightgbm.cv) ---
def lgbm_f1_score_for_cv(preds, train_data):
    labels = train_data.get_label()
    y_pred_binary = (preds > 0.5).astype(int)
    return 'f1_score', f1_score(labels, y_pred_binary, average='weighted'), True


def download_and_process_data(interval, period, end_date, max_retries=5, initial_delay=5):
    logger.info(f"Downloading data for interval {interval} with period {period}...")
    for attempt in range(max_retries):
        try:
            df = yf.download("EURUSD=X", interval=interval, period=period, end=end_date)
            if df.empty:
                logger.warning(f"Empty data received for interval {interval}. No data to process.")
                # Если данные пустые, это может быть не из-за лимита, а просто нет данных.
                # В этом случае не имеет смысла повторять, просто возвращаем None.
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]
            df = df[(df["Close"] > 0) & (df["Open"] > 0)]
            df["Close"] = df["Close"].ffill().bfill()

            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            return df
        except Exception as e:
            # Проверяем, является ли ошибка связана с лимитом запросов
            if "Rate limited" in str(e) or isinstance(e, yf.YFRateLimitError):
                delay = initial_delay * (2**attempt) # Экспоненциальная задержка
                logger.warning(f"Rate limit hit for {interval} data. Retrying in {delay} seconds (attempt {attempt + 1}/{max_retries})...")
                time.sleep(delay)
            else:
                logger.error(f"Error downloading data for interval {interval}: {str(e)}")
                return None # Если другая ошибка, не повторяем
    logger.error(f"Failed to download data for interval {interval} after {max_retries} attempts due to rate limiting.")
    return None


def calculate_indicators(df):
    if df is None or df.empty:
        return pd.DataFrame()
    df_ind = df.copy()
    df_ind["RSI"] = compute_rsi(df_ind["Close"])
    df_ind["MA20"] = df_ind["Close"].rolling(window=20).mean()
    df_ind["BB_Up"], df_ind["BB_Low"] = compute_bollinger_bands(df_ind["Close"])
    df_ind["MACD"], df_ind["MACD_Sig"] = compute_macd(df_ind["Close"])
    df_ind["Stoch_K"], df_ind["Stoch_D"] = compute_stochastic_oscillator(df_ind["High"], df_ind["Low"], df_ind["Close"])
    df_ind["ATR"] = compute_atr(df_ind["High"], df_ind["Low"], df_ind["Close"])
    df_ind["ROC"] = compute_roc(df_ind["Close"])
    df_ind["ADX"] = compute_adx(df_ind["High"], df_ind["Low"], df_ind["Close"])
    df_ind["PSAR"] = compute_psar(df_ind["High"], df_ind["Low"], df_ind["Close"])
    return df_ind

def prepare_data():
    end_date = datetime.now()

    # --- 1. Загрузка данных для основных таймфреймов ---
    df_15m = download_and_process_data("15m", LOOKBACK_PERIOD, end_date)
    if df_15m is None or df_15m.empty:
        raise ValueError("Failed to download or process 15m data.")
    logger.info(f"Downloaded {len(df_15m)} rows for 15m interval.")
    # НОВОЕ: Добавляем небольшую задержку между запросами
    time.sleep(2) 
    # Оставляем только 1-часовые данные для многотаймфреймового анализа
    df_1h = download_and_process_data("1h", "120d", end_date) # 120 дней для 1-часовых данных

    # --- 2. Расчет индикаторов для каждого таймфрейма ---
    df_15m_features = calculate_indicators(df_15m)
    df_1h_features = calculate_indicators(df_1h)

    # --- 3. Объединение признаков разных таймфреймов ---
    df_list = [df_15m_features]

    indicators_to_merge = ["RSI", "MA20", "BB_Up", "BB_Low", "MACD", "MACD_Sig",
                           "Stoch_K", "Stoch_D", "ATR", "ROC", "ADX", "PSAR"]

    # Resample and add multi-timeframe indicators (только для 1h)
    if df_1h_features is not None and not df_1h_features.empty:
        for ind in indicators_to_merge:
            if ind in df_1h_features.columns:
                resampled_df = df_1h_features[[ind]].resample("15min").ffill().rename(columns={ind: f"{ind}_1h"})
                df_list.append(resampled_df)
    
    df_combined = pd.concat(df_list, axis=1)
    df_combined = df_combined[~df_combined.index.duplicated(keep='first')]
    df_combined = df_combined.sort_index()

    # --- 4. Добавление лагированных признаков ---
    lag_periods = [1, 2, 3, 4]
    lagged_features_df = pd.DataFrame(index=df_combined.index)

    cols_to_lag = [col for col in df_combined.columns if col not in ["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"]]
    cols_to_lag.extend(["Close"])

    for col in set(cols_to_lag):
        if col in df_combined.columns:
            for lag in lag_periods:
                lagged_features_df[f"{col}_Lag{lag}"] = df_combined[col].shift(lag)
    
    df_combined = pd.concat([df_combined, lagged_features_df], axis=1)

    # --- 5. Добавление временных признаков ---
    time_features_df = pd.DataFrame(index=df_combined.index)
    time_features_df["Hour"] = df_combined.index.hour
    time_features_df["DayOfWeek"] = df_combined.index.dayofweek
    time_features_df["DayOfMonth"] = df_combined.index.day
    time_features_df["Month"] = df_combined.index.month
    df_combined = pd.concat([df_combined, time_features_df], axis=1)

    # --- 6. Определение целевой переменной ---
    df_combined["Target_Raw"] = df_combined["Close"].shift(-HORIZON_PERIODS)
    
    df_combined["Target"] = np.nan
    df_combined.loc[df_combined["Target_Raw"] > df_combined["Close"] * (1 + TARGET_PRICE_CHANGE_THRESHOLD), "Target"] = 1
    df_combined.loc[df_combined["Target_Raw"] < df_combined["Close"] * (1 - TARGET_PRICE_CHANGE_THRESHOLD), "Target"] = 0

    # --- 7. Очистка данных ---
    df_combined["PriceChange"] = df_combined["Close"].pct_change()
    df_combined = df_combined[df_combined["PriceChange"].abs() < 0.1]
    
    initial_rows = len(df_combined)
    nan_in_target = df_combined["Target"].isna().sum()
    logger.info(f"Rows with NaN in Target before dropna: {nan_in_target}")

    df_combined = df_combined.dropna()
    logger.info(f"After dropna: {len(df_combined)} rows (dropped {initial_rows - len(df_combined)})")

    if len(df_combined) < MIN_DATA_ROWS:
        logger.error(f"Insufficient data: {len(df_combined)} rows, required at least {MIN_DATA_ROWS}.")
        raise ValueError(f"Insufficient data: {len(df_combined)} rows, required at least {MIN_DATA_ROWS}.")
    
    # --- 8. Подготовка X и y ---
    exclude_cols = ["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits", 
                    "Target_Raw", "Target", "PriceChange"]
    
    feature_columns = [col for col in df_combined.columns if col not in exclude_cols and not col.startswith("Adj Close")]
    
    for basic_col in ["Open", "High", "Low", "Close"]:
        if basic_col in df_combined.columns and basic_col not in feature_columns:
            feature_columns.append(basic_col)

    X_raw = df_combined[feature_columns]
    y_raw = df_combined["Target"].astype(int)

    X_raw = X_raw.replace([np.inf, -np.inf], np.nan).ffill().bfill()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X_raw.columns, index=X_raw.index)

    logger.info(f"X shape: {X_scaled_df.shape}, y shape: {y_raw.shape}, y distribution: {y_raw.value_counts().to_dict()}")
    return X_scaled_df, y_raw, scaler, df_combined

# Глобальные переменные для X_train_val, y_train_val, scale_pos_weight
# Они будут установлены в train_model и доступны в objective
X_train_val = None
y_train_val = None
scale_pos_weight = 1.0

def objective(trial):
    global X_train_val, y_train_val, scale_pos_weight # Объявляем, что используем глобальные переменные

    logger.info(f"🚀 Trial #{trial.number} started") # НОВЫЙ ЛОГ
    params = {
        "objective": "binary",
        "metric": "binary_logloss",  # формально нужен, но мы переопределим через feval
        "n_estimators": trial.suggest_int("n_estimators", 50, 300), # Уменьшен верхний предел
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 10, 50), # Уменьшен диапазон
        "max_depth": trial.suggest_int("max_depth", 3, 7), # Уменьшен диапазон
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
        "boosting_type": "gbdt",
        "scale_pos_weight": scale_pos_weight
    }
    
    lgb_train = lgb.Dataset(X_train_val, y_train_val)
    folds = TimeSeriesSplit(n_splits=N_SPLITS_TS_CV) # N_SPLITS_TS_CV из config.py
    
    logger.info(f"Trial #{trial.number}: Starting LightGBM CV with {params['n_estimators']} estimators and {N_SPLITS_TS_CV} folds...") # НОВЫЙ ЛОГ
    start_cv_time = time.time()
    try:
        cv_results = lgb.cv(
            params,
            lgb_train,
            num_boost_round=params["n_estimators"],
            folds=folds,
            feval=lgbm_f1_score_for_cv,
            stratified=False,
            return_cvbooster=False,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                optuna.integration.LightGBMPruningCallback(trial, "valid f1_score") # Исправлено на "valid f1_score"
            ]
        )
        logger.info(f"Trial #{trial.number}: LightGBM CV completed.") # НОВЫЙ ЛОГ
        logger.info(f"⏱ Trial #{trial.number} CV time: {time.time() - start_cv_time:.2f} seconds") # НОВЫЙ ЛОГ

        try:
            avg_f1 = cv_results['cv_agg f1_score-mean'][-1]
        except KeyError:
            # Fallback, если имя метрики отличается (хотя после исправлений не должно быть)
            avg_f1 = cv_results['cv_agg f1_score-mean'][-1] 
            logger.warning(f"Trial #{trial.number}: Used fallback metric name for F1-score.")
            
        logger.info(f"Trial #{trial.number}: Average F1-score = {avg_f1:.4f}") # НОВЫЙ ЛОГ
        
        return avg_f1

    except Exception as e:
        logger.error(f"Error during LightGBM CV for trial #{trial.number}: {e}")
        # Если произошла ошибка во время CV, можно прунить пробный запуск
        raise TrialPruned() # Это сообщит Optuna, что пробный запуск неудачен

def train_model():
    global X_train_val, y_train_val, scale_pos_weight # Объявляем, что будем изменять глобальные переменные

    logger.info("Starting model training process...")
    try:
        X, y, scaler, df_original = prepare_data()
    except Exception as e:
        logger.error(f"Data preparation error, cannot train model: {e}")
        return

    if len(X) != len(y):
        logger.error("X and y must have the same number of samples.")
        return

    X_train_val_local, X_test, y_train_val_local, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False, stratify=None
    )
    
    # Присваиваем глобальным переменным значения из локальных
    X_train_val = X_train_val_local
    y_train_val = y_train_val_local

    logger.info(f"Train/Validation set size: {len(X_train_val)}, Test set size: {len(X_test)}")

    neg_count = y_train_val.value_counts().get(0, 0)
    pos_count = y_train_val.value_counts().get(1, 0)
    
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    logger.info(f"Class distribution in training data: Neg={neg_count}, Pos={pos_count}. Scale_pos_weight={scale_pos_weight}")

    logger.info("🔍 Starting Optuna hyperparameter search for LightGBM...")
    logger.info(f"Optuna storage URL: {OPTUNA_STORAGE_URL}") # Новый лог
    logger.info(f"Optuna study name: {OPTUNA_STUDY_NAME}") # Новый лог

    try:
        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.HyperbandPruner(),
            sampler=optuna.samplers.TPESampler(),
            study_name=OPTUNA_STUDY_NAME,
            storage=OPTUNA_STORAGE_URL,
            load_if_exists=True
        )
        logger.info("Optuna study created/loaded successfully.") # Новый лог
    except Exception as e:
        logger.error(f"Error creating/loading Optuna study: {e}")
        return

    try:
        study.optimize(objective, n_trials=None, timeout=MAX_TRAINING_TIME)
        logger.info("Optuna optimization completed.") # Новый лог
    except Exception as e:
        logger.error(f"Error during Optuna optimization: {e}")
        return

    best_params = study.best_params
    best_f1_score = study.best_value

    logger.info(f"✅ Optuna best F1-score: {best_f1_score:.4f}")
    logger.info(f"📋 Best params: {best_params}")

    if best_f1_score < MIN_ACCURACY_FOR_SIGNAL:
        logger.warning(f"❌ F1-score {best_f1_score:.2f} too low. No model saved.")
        return

    final_model = LGBMClassifier(**best_params, random_state=42, n_jobs=-1, verbose=-1)
    final_model.fit(X_train_val, y_train_val)
    logger.info("🎯 Финальное обучение модели завершено")

    y_pred_test = final_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_f1_score = f1_score(y_test, y_pred_test, average='weighted')
    logger.info(f"Final model performance on TEST set: Accuracy={test_accuracy:.4f}, F1-score={test_f1_score:.4f}")

    joblib.dump(final_model, MODEL_PATH)
    joblib.dump({'scaler': scaler}, 'scaler.pkl')

    with open(ACCURACY_PATH, "w") as f:
        json.dump({
            "accuracy": float(test_accuracy),
            "f1_score": float(test_f1_score),
            "last_trained": str(datetime.now()),
            "best_params": best_params
        }, f)

    if len(X) > 0:
        # Для генерации сигнала нужно получить последние признаки
        # df_original теперь содержит все признаки, включая многотаймфреймовые и лаги
        # и уже очищен от NaN.
        # Берем последнюю строку из X (масштабированные признаки)
        latest_features_scaled = X.iloc[-1:].values
        # Берем последнюю строку из df_original (исходные данные + все признаки)
        latest_original_data_point = df_original.iloc[-1:]

        generate_signal(final_model, scaler, latest_features_scaled, latest_original_data_point)
    else:
        logger.warning("No data to generate signal after training.")

def generate_signal(model, scaler, latest_features_scaled, latest_original_data_point):
    try:
        if latest_features_scaled.shape[0] == 0:
            logger.warning("No latest features data to generate signal.")
            return
        if latest_original_data_point.empty:
            logger.warning("No latest original data point to get current price.")
            return

        prediction_proba = model.predict_proba(latest_features_scaled)[0]
        
        current_price = latest_original_data_point['Close'].iloc[0] 
        
        signal_type = "HOLD"
        stop_loss = None
        take_profit = None

        buy_probability = prediction_proba[1]
        sell_probability = prediction_proba[0]

        if buy_probability >= PREDICTION_PROB_THRESHOLD:
            signal_type = "BUY"
            # Пример расчета SL/TP (можно настроить)
            stop_loss = current_price * (1 - 0.001) # 0.1% ниже текущей цены
            take_profit = current_price * (1 + 0.002) # 0.2% выше текущей цены
        elif sell_probability >= PREDICTION_PROB_THRESHOLD:
            signal_type = "SELL"
            stop_loss = current_price * (1 + 0.001) # 0.1% выше текущей цены
            take_profit = current_price * (1 - 0.002) # 0.2% ниже текущей цены

        message = (
            f"📈 Trading Signal: {signal_type}\n"
            f"Current Price: {current_price:.5f}\n"
            f"Buy Probability: {buy_probability:.4f}\n"
            f"Sell Probability: {sell_probability:.4f}"
        )
        if stop_loss and take_profit:
            message += f"\nStop Loss: {stop_loss:.5f}\nTake Profit: {take_profit:.5f}"

        logger.info(message)
        send_telegram_message(message)

    except Exception as e:
        logger.error(f"Error generating signal: {e}")


@app.on_event("startup")
def startup_event():
    Thread(target=train_model).start()

@app.get("/")
async def root():
    # Проверяем, существует ли модель и файл с точностью
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ACCURACY_PATH):
        logger.info("Model or accuracy file not found. Training process should be running in background.")
        return {"status": "Model training in progress or not yet started."}

    try:
        with open(ACCURACY_PATH, "r") as f:
            accuracy_data = json.load(f)
        current_f1_score = accuracy_data.get("f1_score", 0.0)
        last_trained = datetime.fromisoformat(accuracy_data["last_trained"])

        # Проверяем, нужно ли переобучать модель
        # Если прошло более 1 дня или точность ниже целевой
        if (datetime.now() - last_trained).days >= 1 or current_f1_score < TARGET_ACCURACY:
            logger.info(f"Model needs retraining. Last trained: {last_trained}, F1-score: {current_f1_score:.4f}. Starting retraining in background.")
            # Запускаем обучение в отдельном потоке, чтобы не блокировать основной поток FastAPI
            Thread(target=train_model).start()
            return {"status": "Model retraining initiated in background."}

        # Если модель актуальна и достаточно точна, генерируем сигнал
        if current_f1_score >= MIN_ACCURACY_FOR_SIGNAL:
            model = joblib.load(MODEL_PATH)
            scaler_data = joblib.load('scaler.pkl')
            scaler = scaler_data['scaler']
            
            end_date = datetime.now()
            df_latest_full_15m = download_and_process_data("15m", "7d", end_date)
            if df_latest_full_15m is None:
                logger.error("Failed to download 15m data for signal generation.")
                return {"status": "Error: Could not download latest 15m data for signal."}

            time.sleep(2) # Задержка между запросами

            df_latest_full_1h = download_and_process_data("1h", "120d", end_date)
            if df_latest_full_1h is None:
                logger.error("Failed to download 1h data for signal generation.")
                return {"status": "Error: Could not download latest 1h data for signal."}

            # Расчет индикаторов для последних данных
            df_latest_full_15m_features = calculate_indicators(df_latest_full_15m)
            df_latest_full_1h_features = calculate_indicators(df_latest_full_1h)

            # Объединение признаков для последних данных
            df_list_latest = [df_latest_full_15m_features]
            indicators_to_merge = ["RSI", "MA20", "BB_Up", "BB_Low", "MACD", "MACD_Sig",
                                   "Stoch_K", "Stoch_D", "ATR", "ROC", "ADX", "PSAR"]

            if df_latest_full_1h_features is not None and not df_latest_full_1h_features.empty:
                for ind in indicators_to_merge:
                    if ind in df_latest_full_1h_features.columns:
                        resampled_df = df_latest_full_1h_features[[ind]].resample("15min").ffill().rename(columns={ind: f"{ind}_1h"})
                        df_list_latest.append(resampled_df)

            df_combined_latest = pd.concat(df_list_latest, axis=1)
            df_combined_latest = df_combined_latest[~df_combined_latest.index.duplicated(keep='first')]
            df_combined_latest = df_combined_latest.sort_index()

            # Добавление лагированных признаков для последних данных
            lag_periods = [1, 2, 3, 4]
            lagged_features_df_latest = pd.DataFrame(index=df_combined_latest.index)
            cols_to_lag_latest = [col for col in df_combined_latest.columns if col not in ["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"]]
            cols_to_lag_latest.extend(["Close"])

            for col in set(cols_to_lag_latest):
                if col in df_combined_latest.columns:
                    for lag in lag_periods:
                        lagged_features_df_latest[f"{col}_Lag{lag}"] = df_combined_latest[col].shift(lag)
            df_combined_latest = pd.concat([df_combined_latest, lagged_features_df_latest], axis=1)

            # Добавление временных признаков для последних данных
            df_combined_latest["Hour"] = df_combined_latest.index.hour
            df_combined_latest["DayOfWeek"] = df_combined_latest.index.dayofweek
            df_combined_latest["DayOfMonth"] = df_combined_latest.index.day
            df_combined_latest["Month"] = df_combined_latest.index.month

            # Очистка последних данных (аналогично prepare_data)
            df_combined_latest["PriceChange"] = df_combined_latest["Close"].pct_change()
            df_combined_latest = df_combined_latest[df_combined_latest['PriceChange'].abs() < 0.1]
            df_combined_latest = df_combined_latest.dropna()

            if len(df_combined_latest) >= 1:
                # Исключаем колонки, которые не являются признаками (аналогично prepare_data)
                exclude_cols_signal = ["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits", 
                                       "Target_Raw", "Target", "PriceChange"]
                
                feature_columns_signal = [col for col in df_combined_latest.columns if col not in exclude_cols_signal and not col.startswith("Adj Close")]
                
                for basic_col in ["Open", "High", "Low", "Close"]:
                    if basic_col in df_combined_latest.columns and basic_col not in feature_columns_signal:
                        feature_columns_signal.append(basic_col)

                latest_features_raw = df_combined_latest[feature_columns_signal].iloc[-1:]
                latest_features_scaled = scaler.transform(latest_features_raw)
                
                latest_original_data_point = df_combined_latest.iloc[-1:]

                generate_signal(model, scaler, latest_features_scaled, latest_original_data_point)
            else:
                logger.warning("Could not get enough latest data to generate signal from existing model (need at least 1 row after feature engineering).")
        else:
            logger.warning(f"Model F1-score {current_f1_score:.4f} is below MIN_ACCURACY_FOR_SIGNAL ({MIN_ACCURACY_FOR_SIGNAL}). No signal generated.")
            return {"status": "Model accuracy too low for signal generation."}

    except Exception as e:
        logger.error(f"Error loading model or generating signal from existing model: {e}")
        return {"status": f"Error: {e}"}

    return {"status": "Bot is running"}

if __name__ == "__main__":
    uvicorn.run(app, host=config.UVICORN_HOST, port=config.UVICORN_PORT)
