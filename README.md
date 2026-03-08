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

How it works: 
1. Load and preprocess data
   - Converts Date column to datetime
   - Sort by date
   - Create day index for modeling
   - Calculate technical indicators and engineered features
   - Build lag features for OHLCV
2. Target creation
   - Predict next-day return:
     data["target_return"] = data["Close"].pct_change().shift(-1)
   - Optional direction feature:
     data["direction"] = (data["Close"].pct_change() > 0).astype(int)
3. Train/Test Split
   - 80% for training, 20% for testing
   - Features include lagged OHLCV, technical indicators, and engineered features
   - Target = target_return
4. Train Model
   - Uses XGBoost Regressor
   - Hyperparameters optimized for stability and performance
5. Evaluate Model
   - MAE of returns → average prediction error
   - Direction accuracy → how often the model predicts up vs down correctly
   - Return correlation → correlation between predicted and actual return
6. 7-Day Forecast
   - Recursive forecast using predicted returns to compute future Close prices
   - Skips weekends using business days
   - Saves results to Excel (COVER Corporation_next_7_days.xlsx)

Note: The MAE is about 4%. I am still working on getting it close to 0%

Results:
Apple
Test MAE Return: 0.011828 (1.18%)
Direction Accuracy: 58.54%
Return Correlation: 0.160

ANYCOLOR Inc
Test MAE Return: 0.024686 (2.46%)
Direction Accuracy: 23.81%
Return Correlation: 0.497

COVER Corporation
Test MAE Return: 0.020955 (2.09%)
Direction Accuracy: 65.08%
Return Correlation: 0.316
