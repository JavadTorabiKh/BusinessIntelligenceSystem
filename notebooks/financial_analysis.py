import pandas as pd

# Load data
df = pd.read_csv("../data/financial_data.csv")

# Quick analysis
print("Data Summary:")
print(df.groupby(["Type", "Department"])["Amount"].sum().unstack())

# Create pivot table
report = df.pivot_table(
    index="Department", columns="Type", values="Amount", aggfunc="sum", margins=True
)
print("\nFinancial Report by Department:")
print(report)


# ==========================


from statsmodels.tsa.arima.model import ARIMA

# 1. Prepare monthly data
monthly_data = df.groupby(pd.to_datetime(df["Date"]).dt.to_period("M"))["Amount"].sum()

# ARIMA Model
model = ARIMA(monthly_data, order=(1, 1, 1))
results = model.fit()

# 3-month forecast
forecast = results.forecast(steps=3)
print("3-Month Forecast:")
print(forecast)


# ==========================


import numpy as np

# 2. Profitability Analysis

df["Profit"] = np.where(df["Type"] == "Revenue", df["Amount"], -df["Amount"])
profitability = df.groupby("Department")["Profit"].sum().sort_values()

profitability


# ==========================


# 3. Time Series Trend

monthly_trend = df.groupby(pd.to_datetime(df["Date"]).dt.to_period("M"))["Amount"].sum()
monthly_trend.plot(title="Monthly Financial Trend")


# ==========================


# 4. Top Performers

top_depts = (
    df[df["Type"] == "Revenue"].groupby("Department")["Amount"].sum().nlargest(3)
)
top_depts


# ==========================


# 5. Anomaly Detection

from sklearn.ensemble import IsolationForest

model = IsolationForest(contamination=0.05)
df["Anomaly"] = model.fit_predict(df[["Amount"]])
anomalies = df[df["Anomaly"] == -1]
anomalies


# ==========================


# 6. Financial Forecasting

from prophet import Prophet
import pandas as pd

monthly_trend = df.groupby(pd.to_datetime(df["Date"]).dt.to_period("M"))["Amount"].sum()

forecast_data = monthly_trend.reset_index()
forecast_data["Date"] = forecast_data["Date"].dt.to_timestamp()
forecast_data.columns = ["ds", "y"]

model = Prophet()
model.fit(forecast_data)

future = model.make_future_dataframe(periods=6, freq="M")
forecast = model.predict(future)

fig = model.plot(forecast)


# ==========================


# 7. Cost Allocation Analysis

cost_ratio = (
    df[df["Type"] == "Expense"].groupby("Department")["Amount"].sum()
)  # total_expenses
cost_ratio


# ==========================


# 8. Auto-Tagging

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["Description"])
kmeans = KMeans(n_clusters=5)
df["Category"] = kmeans.fit_predict(X)


# ==========================


# 9. Cash Flow Cycle

cash_flow = (
    df.groupby([pd.to_datetime(df["Date"]).dt.dayofyear, "Type"])["Amount"]
    .sum()
    .unstack()
)
cash_flow


# ==========================


# 10. Interactive Reports

import altair as alt
import streamlit as st

st.title("Financial Dashboard")
chart = (
    alt.Chart(df)
    .mark_bar()
    .encode(
        x="Department",
        y="sum(Amount)",
        color="Type",
        tooltip=["Department", "sum(Amount)"],
    )
    .interactive()
)


st.altair_chart(chart, use_container_width=True)
