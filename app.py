import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from keras.models import load_model
import streamlit as st
from sentiment import Newsanalysis

news = Newsanalysis()


st.set_page_config(
    page_title="Stock prediction app",
    page_icon="market-analysis.png"
)

st.title("💹 Stock Trend Prediction")

start = '2010-01-01'
end = st.text_input("📅 Enter Current Date", '2023-07-29')

user_input = st.text_input("Enter Stock Ticker", "WIT")

# ─────────────────────────────────────────────
# Market News Sentiment
# ─────────────────────────────────────────────
st.subheader("🤔 Market News Sentiment")
with st.spinner("Fetching latest news…"):
    result = news.newsa(user_input)
if isinstance(result, str):
    st.warning(result)
else:
    st.dataframe(result)

# ─────────────────────────────────────────────
# Employer / Employee Sentiment
# ─────────────────────────────────────────────
st.subheader("👥 Employer & Employee Sentiment")
with st.spinner("Fetching employer data…"):
    employer = news.get_employer_sentiment(user_input)

if employer:
    col1, col2, col3 = st.columns(3)
    col1.metric("🏢 Company", employer.get("company", "N/A"))
    col2.metric("👷 Employees", str(employer.get("employees", "N/A")))
    col3.metric("🏭 Sector", employer.get("sector", "N/A"))

    if "glassdoor_rating" in employer:
        rating = employer["glassdoor_rating"]
        sentiment = employer["sentiment"]
        color = "🟢" if sentiment == "Positive" else ("🔴" if sentiment == "Negative" else "🟡")
        st.metric(
            f"{color} Glassdoor Rating (out of 5)",
            f"{rating:.1f}",
            delta=sentiment,
        )
    else:
        st.info(
            f"ℹ️ Glassdoor data unavailable. "
            f"Industry: **{employer.get('industry', 'N/A')}** | "
            f"Source: {employer.get('source', 'N/A')}"
        )
else:
    st.warning("Could not fetch employer sentiment data.")

# ─────────────────────────────────────────────
# Future Plans & Analyst Outlook
# ─────────────────────────────────────────────
st.subheader("🔭 Future Plans & Analyst Outlook")
with st.spinner("Fetching forward-looking data…"):
    future = news.get_future_plans(user_input)

if future:
    # Key forward-looking metrics
    col1, col2, col3 = st.columns(3)
    rec = future.get("recommendation", "N/A")
    col1.metric("📊 Analyst Recommendation", str(rec).upper())

    target = future.get("target_mean_price", "N/A")
    col2.metric("🎯 Mean Price Target", f"${target}" if target != "N/A" else "N/A")

    rev_growth = future.get("revenue_growth", "N/A")
    if rev_growth != "N/A" and rev_growth is not None:
        col3.metric("📈 Revenue Growth", f"{rev_growth * 100:.1f}%")
    else:
        col3.metric("📈 Revenue Growth", "N/A")

    col4, col5 = st.columns(2)
    fpe = future.get("forward_pe", "N/A")
    col4.metric("💡 Forward P/E", f"{fpe:.1f}" if isinstance(fpe, float) else str(fpe))

    eg = future.get("earnings_growth", "N/A")
    if eg != "N/A" and eg is not None:
        col5.metric("💰 Earnings Growth", f"{eg * 100:.1f}%")
    else:
        col5.metric("💰 Earnings Growth", "N/A")

    # Earnings / upcoming events
    if future.get("earnings_dates") is not None:
        st.write("**📅 Upcoming Earnings Calendar**")
        st.dataframe(future["earnings_dates"])

    # Analyst recommendations history
    if future.get("analyst_recommendations") is not None:
        st.write("**🔬 Recent Analyst Recommendations**")
        st.dataframe(future["analyst_recommendations"])

    # Business summary
    summary = future.get("business_summary", "")
    if summary:
        with st.expander("📋 Business Summary & Strategy"):
            st.write(summary)
else:
    st.warning("Could not fetch future plans data.")

# ─────────────────────────────────────────────
# Historical Price Data
# ─────────────────────────────────────────────
df = yf.download(user_input, start=start, end=end)

st.subheader(f"📅 Data from {start} - {end}")
st.write(df.describe())

st.subheader("💸 Closing Price VS Time Chart")
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader("💸 Closing Price VS Time Chart with MA 100")
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100, "r")
plt.plot(df.Close)
st.pyplot(fig)

st.subheader("💸 Closing Price VS Time Chart with MA100 & MA200")
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100, "r")
plt.plot(ma200, "g")
plt.plot(df.Close, "b")
st.pyplot(fig)

data_training = pd.DataFrame(df["Close"][0:int(len(df) * 0.70)])
data_testing = pd.DataFrame(df["Close"][int(len(df) * 0.70): int(len(df))])

scaler = MinMaxScaler(feature_range=(0, 1))
data_traning_array = scaler.fit_transform(data_training)

# ─────────────────────────────────────────────
# LSTM Model Predictions
# ─────────────────────────────────────────────
model = load_model('keras.model.h5')

past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i - 100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

y_predicted = model.predict(x_test)

scaler = scaler.scale_
scale_factor = 1 / scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

st.subheader("🤓 Predictions VS Original")
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, "b", label="Original Price")
plt.plot(y_predicted, "r", label="Predicted Price")
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)






























