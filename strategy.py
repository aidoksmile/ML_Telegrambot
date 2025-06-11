from backtesting import Strategy

class MLStrategy(Strategy):
    def init(self):
        pass

    def next(self):
        if len(self.data) < 2:
            return

        model = self._broker._model
        features = self._broker._features

        try:
            prediction = model.predict(features.iloc[-1:].values)[0]
        except Exception as e:
            print(f"Ошибка прогнозирования: {e}")
            return

        price = self.data.Close[-1]
        sl = price - 2 * self.data.volatility[-1]
        tp = price + 4 * self.data.volatility[-1]

        if prediction == 1 and not self.position:
            self.buy(size=0.1, sl=sl, tp=tp)
        elif prediction == 0 and self.position:
            self.position.close()
