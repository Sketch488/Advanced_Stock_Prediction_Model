import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

# Config
file = "COVER Corporation.xlsx"
forecast_days = 7
n_lags = 10
target_cols = ["Open", "High", "Low", "Close", "Volume"]

# Load and prepare
data = pd.read_excel(file)
data["Date"] = pd.to_datetime(data["Date"])
data = data.sort_values("Date").reset_index(drop=True)
data["Days"] = range(len(data)) 

# MAES
data["MA10"] = data["Close"].rolling(10).mean()
data["MA50"] = data["Close"].rolling(50).mean()

# Volatility
data["Volatility"] = data["Close"].pct_change().rolling(10).std()

# Percentage Change

# Price Difference

# Technical Indicators

# Date Features

# Interaction Features

# Build lag for all columns
for col in target_cols:
    for i in range(1, n_lags + 1):
        data[f"{col}_lag_{i}"] = data[col].shift(i)

# Build features
feature_cols = ["Days", "MA10", "MA50", "Volatility"] + [
    f"{col}_lag_{i}" for col in target_cols for i in range(1, n_lags + 1)
]
model_data = data.dropna().reset_index(drop=True)

X = model_data[feature_cols]
Y = model_data[target_cols]

# 80% training and 20% testing
split = int(len(model_data) * 0.8)
X_train = X.iloc[:split]
X_test = X.iloc[split:]
Y_train = Y.iloc[:split]
Y_test = Y.iloc[split:]

# Train model
base_model = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1,
    random_state=42,
)
model = MultiOutputRegressor(base_model)
model.fit(X_train, Y_train)

# Accuracy evaluation
if len(X_test) > 0:
    test_pred = model.predict(X_test)
    maes = mean_absolute_error(Y_test, test_pred, multioutput="raw_values")
    for col, score in zip(target_cols, maes):
        print(f"Test MAE {col}: {score:.2f}")

# Stores history of actual + predicted for feeding lags in next steps
history = model_data[["Date", "Days"] + target_cols].copy()
last_date = history["Date"].iloc[-1]
last_day = int(history["Days"].iloc[-1])
rows = []

# Predict next days
for step in range(1, forecast_days + 1):
    next_day = last_day + step
    
    # Dynamic lag input
    feature_row = {"Days": next_day}
    feature_row["MA10"] = history["Close"].tail(10).mean()
    feature_row["MA50"] = history["Close"].tail(50).mean()
    feature_row["Volatility"] = history["Close"].pct_change().tail(10).std()
    for col in target_cols:
        for i in range(1, n_lags + 1):
            feature_row[f"{col}_lag_{i}"] = history[col].iloc[-i]

    X_next = pd.DataFrame([feature_row], columns=feature_cols)
    pred = model.predict(X_next)[0]
    
    # Skips weekends
    next_date = last_date + pd.offsets.BDay(step)
    pred_row = {"Date": next_date}
    for idx, col in enumerate(target_cols):
        pred_row[col] = float(pred[idx])

    rows.append(pred_row)

    # Feed predictions back as lag inputs for subsequent steps.
    history = pd.concat(
        [
            history,
            pd.DataFrame(
                [
                    {
                        "Date": next_date,
                        "Days": next_day,
                        "Open": pred_row["Open"],
                        "High": pred_row["High"],
                        "Low": pred_row["Low"],
                        "Close": pred_row["Close"],
                        "Volume": pred_row["Volume"],
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

# Format
pred_df = pd.DataFrame(rows)
for col in ["Open", "High", "Low", "Close"]:
    pred_df[col] = pred_df[col].round(0).astype("Int64")
pred_df["Volume"] = pred_df["Volume"].round(0).astype("Int64")
pred_df["Date"] = pred_df["Date"].dt.strftime("%Y-%m-%d")

# To Excel
output_file = file.replace(".xlsx", "_next_7_days.xlsx")
pred_df.to_excel(output_file, index=False)
print(pred_df.to_string(index=False))
print(f"Saved: {output_file}")
