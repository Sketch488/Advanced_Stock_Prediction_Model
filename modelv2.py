import pandas as pd
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import ta

# Config
file = "COVER Corporation.xlsx"
forecast_days = 7
n_lags = 10
target_col = "target_return"

# Load and prepare
data = pd.read_excel(file)
data["Date"] = pd.to_datetime(data["Date"])
data = data.sort_values("Date").reset_index(drop=True)
data["Days"] = range(len(data))

# Moving Averages
data["MA10"] = data["Close"].rolling(10).mean()
data["MA50"] = data["Close"].rolling(50).mean()

# Volatility
data["Volatility"] = data["Close"].pct_change().rolling(10).std()

# Percentage Change
data["return_1"] = data["Close"].pct_change()
data["return_5"] = data["Close"].pct_change(5)

# Price Difference
data["close_diff"] = data["Close"] - data["Open"]

# Technical Indicators
data["rsi"] = ta.momentum.RSIIndicator(data["Close"]).rsi()
data["macd"] = ta.trend.MACD(data["Close"]).macd()

# Date Features
data["day_of_week"] = data["Date"].dt.dayofweek
data["month"] = data["Date"].dt.month

# Interaction Features
data["high_low_diff"] = data["High"] - data["Low"]
data["volume_price"] = data["Volume"] * data["Close"]

# Build lag features for OHLCV
lag_cols = ["Open", "High", "Low", "Close", "Volume"]

# Returns
data["target_return"] = data["Close"].pct_change().shift(-1)

data["direction"] = (data["Close"].pct_change() > 0).astype(int)

for col in lag_cols:
    for i in range(1, n_lags + 1):
        data[f"{col}_lag_{i}"] = data[col].shift(i)

# Feature list
feature_cols = [
    "Days",
    "MA10",
    "MA50",
    "Volatility",
    "return_1",
    "return_5",
    "close_diff",
    "rsi",
    "macd",
    "day_of_week",
    "month",
    "high_low_diff",
    "volume_price",
    "direction",
] + [
    f"{col}_lag_{i}" for col in lag_cols for i in range(1, n_lags + 1)
]

# Prepare dataset
model_data = data.dropna().reset_index(drop=True)

X = model_data[feature_cols]
Y = model_data[target_col]

# 80% training and 20% testing
split = int(len(model_data) * 0.8)
X_train = X.iloc[:split]
X_test = X.iloc[split:]
Y_train = Y.iloc[:split]
Y_test = Y.iloc[split:]

# Train model
model = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=800,
    learning_rate=0.03,
    max_depth=4,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1,
    random_state=42,
)

model.fit(X_train, Y_train)

# Accuracy evaluation
if len(X_test) > 0:
    test_pred = model.predict(X_test)

    # MAE
    mae = mean_absolute_error(Y_test, test_pred)
    print(f"Test MAE Return: {mae:.6f}")

    # Direction accuracy
    actual_direction = Y_test.values > 0
    pred_direction = test_pred > 0
    direction_accuracy = (actual_direction == pred_direction).mean()

    print(f"Direction Accuracy: {direction_accuracy * 100:.2f}%")

    # Return correlation
    import numpy as np
    actual_returns = np.diff(Y_test.values)
    pred_returns = np.diff(test_pred)
    correlation = np.nan_to_num(np.corrcoef(actual_returns, pred_returns)[0, 1])
    print(f"Return Correlation: {correlation:.3f}")

# History for recursive forecasting
history = model_data[["Date", "Days", "Open", "High", "Low", "Close", "Volume"]].copy()

last_date = history["Date"].iloc[-1]
last_day = int(history["Days"].iloc[-1])
rows = []

# Predict next days
for step in range(1, forecast_days + 1):

    next_day = last_day + step
    next_date = last_date + pd.offsets.BDay(step)

    feature_row = {
        "Days": next_day,
        "MA10": history["Close"].tail(10).mean(),
        "MA50": history["Close"].tail(50).mean(),
        "Volatility": history["Close"].pct_change().tail(10).std(),
        "return_1": history["Close"].pct_change().iloc[-1],
        "return_5": history["Close"].pct_change(5).iloc[-1],
        "close_diff": history["Close"].iloc[-1] - history["Open"].iloc[-1],
        "day_of_week": next_date.dayofweek,
        "month": next_date.month,
        "high_low_diff": history["High"].iloc[-1] - history["Low"].iloc[-1],
        "volume_price": history["Volume"].iloc[-1] * history["Close"].iloc[-1],
    }

    # Indicators recomputed from history
    temp_close = history["Close"]
    feature_row["rsi"] = ta.momentum.RSIIndicator(temp_close).rsi().iloc[-1]
    feature_row["macd"] = ta.trend.MACD(temp_close).macd().iloc[-1]

    # Lag inputs
    for col in lag_cols:
        for i in range(1, n_lags + 1):
            feature_row[f"{col}_lag_{i}"] = history[col].iloc[-i]

    X_next = pd.DataFrame([feature_row], columns=feature_cols)

    pred_return = model.predict(X_next)[0]
    current_close = history["Close"].iloc[-1]

    pred_close = current_close * (1 + pred_return)

    pred_row = {
        "Date": next_date,
        "Close": float(pred_close)
    }

    rows.append(pred_row)

    # Append predicted close for next step
    history = pd.concat(
        [
            history,
            pd.DataFrame(
                [{
                    "Date": next_date,
                    "Days": next_day,
                    "Open": history["Open"].iloc[-1],
                    "High": history["High"].iloc[-1],
                    "Low": history["Low"].iloc[-1],
                    "Close": pred_row["Close"],
                    "Volume": history["Volume"].iloc[-1],
                }]
            ),
        ],
        ignore_index=True,
    )

# Format output
pred_df = pd.DataFrame(rows)
pred_df["Close"] = pred_df["Close"].round(0).astype("Int64")
pred_df["Date"] = pred_df["Date"].dt.strftime("%Y-%m-%d")

# Save to Excel
output_file = file.replace(".xlsx", "_next_7_days.xlsx")
pred_df.to_excel(output_file, index=False)

print(pred_df.to_string(index=False))
print(f"Saved: {output_file}")