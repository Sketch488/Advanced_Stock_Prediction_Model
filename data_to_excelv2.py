import os
import pandas as pd
import yfinance as yf

tickers = ["5253.T", "5032.T"]
company_names = ["COVER Corporation", "ANYCOLOR Inc"]
fields = ["Open", "High", "Low", "Close", "Volume"]

data = yf.download(tickers, period="6mo", interval="1d")

def write_company_file(ticker, company_name):
    filepath = f"{company_name}.xlsx"
    company_df = data.loc[:, (fields, ticker)].copy()
    company_df.columns = fields
    company_df = company_df.reset_index()

    if os.path.exists(filepath):
        existing = pd.read_excel(filepath)
        combined = (
            existing.set_index("Date")
            .combine_first(company_df.set_index("Date"))
            .reset_index()
        )
        combined = (
            combined.drop_duplicates(subset="Date", keep="first")
            .dropna(subset=["Date"] + fields)
            .reset_index(drop=True)
        )
    else:
        combined = company_df

    for col in ["Open", "High", "Low", "Close"]:
        combined[col] = combined[col].round(0).astype("Int64")
        combined["Volume"] = combined["Volume"].round(0).astype("Int64")
        combined["Date"] = pd.to_datetime(combined["Date"])
        combined = combined.sort_values(by="Date", ascending=False).reset_index(drop=True)
        combined["Date"] = combined["Date"].dt.strftime("%Y-%m-%d")

        combined.to_excel(filepath, index=False)
    return combined

for ticker, name in zip(tickers, company_names):
    write_company_file(ticker, name)
