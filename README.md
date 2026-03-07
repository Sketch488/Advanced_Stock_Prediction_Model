This project predicts the next-day stock returns for a given stock (here, COVER Corporation) using historical OHLCV data and technical indicators.
The model uses XGBoost and a set of engineered features including:
1. Moving averages (MA10, MA50)
2. Volatility
3. Technical Indicators (RSI, MACD)
4. Price and volume indicators
5. Lag features for Open, High, Low, Close, Volume

Model also calculates:
1. Mean Absolute Error (MAE) of returns
2. Direction accuracy (Up or down)
3. Return correlation

Libraries:
1. Pandas
2. Numpy
3. XGBoost
4. ta
5. openyxl
