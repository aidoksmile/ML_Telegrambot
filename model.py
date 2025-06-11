            raise ValueError("DataFrame пуст после удаления NaN для признаков")

        features = ['open', 'high', 'low', 'close', 'volume', 'ma_short', 'ma_long', 'volatility', 'momentum']
        X = df[features]
        y = df['direction']

        logging.info(f"Признаки подготовлены: X.shape={X.shape}, y.shape={y.shape}")
        return X, y
    except Exception as e:
        logging.error(f"Ошибка подготовки признаков: {e}")
        raise Exception(f"Ошибка подготовки признаков: {e}")

def optimize_model_params(X, y):
    logging.info("Оптимизация параметров модели")
    try:
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }

        grid_search = GridSearchCV(
            estimator=RandomForestClassifier(),
            param_grid=param_grid,
            scoring='accuracy',
            cv=3,
            n_jobs=-1,
            verbose=0
        )

        grid_search.fit(X, y)
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        logging.info(f"Оптимизация завершена: best_params={best_params}, best_score={best_score * 100:.2f}%")
        return grid_search.best_estimator_, best_score
    except Exception as e:
        logging.error(f"Ошибка оптимизации модели: {e}")
        raise Exception(f"Ошибка оптимизации модели: {e}")

def train_model(X, y):
    logging.info("Обучение модели")
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(X_test))
        logging.info(f"Точность модели: {accuracy * 100:.2f}%")

        if accuracy >= 0.8:
            logging.info("Модель прошла порог точности")
            return model

        logging.warning("Точность ниже 80%, запуск оптимизации...")
        optimized_model, _ = optimize_model_params(X, y)
        acc = accuracy_score(y_test, optimized_model.predict(X_test))
        logging.info(f"Точность оптимизированной модели: {acc * 100:.2f}%")

        if acc < 0.8:
            logging.error(f"Не удалось достичь точности 80%: {acc * 100:.2f}%")
            raise ValueError(f"Не удалось достичь точности 80%: {acc * 100:.2f}%")

        return
