# stock_predictor_app.py

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

# Helper: Load data
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data[['Close']]

# Helper: Preprocess for LSTM
def preprocess_data(data, window_size):
    close_prices = data['Close'].values.reshape(-1, 1)

    X, y = [], []
    for i in range(window_size, len(close_prices)):
        X.append(close_prices[i-window_size:i])
        y.append(close_prices[i])

    X = np.array(X)
    y = np.array(y)

    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X.reshape(-1, 1)).reshape(X.shape)
    y_scaled = scaler.fit_transform(y)

    return X_scaled, y_scaled, scaler

# Helper: Build the LSTM model
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, activation='tanh', input_shape=input_shape, return_sequences=True))
    model.add(LSTM(32, activation='tanh'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Helper: Predict future prices
def predict_future(model, recent_data, days_to_predict, scaler):
    predictions = []
    current_input = recent_data[-1].copy()

    for _ in range(days_to_predict):
        pred = model.predict(current_input.reshape(1, *current_input.shape), verbose=0)
        predictions.append(pred[0][0])
        current_input = np.append(current_input[1:], [[pred[0][0]]], axis=0)

    predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    return predicted_prices

# -------------------- STREAMLIT APP --------------------
st.title("📈 Stock Price Predictor using LSTM")

ticker = st.text_input("Enter Stock Ticker (e.g. RELIANCE.NS)", value="RELIANCE.NS")
train_years = st.slider("Training Duration (Years)", 1, 10, 5)
predict_days = st.slider("Number of Days to Predict", 1, 100, 30)

if st.button("Predict"):
    try:
        end_date = pd.Timestamp.today()
        start_date = end_date - pd.DateOffset(years=train_years)

        st.info(f"Fetching data for {ticker} from {start_date.date()} to {end_date.date()}...")
        data = load_data(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

        if data.empty or len(data) < 200:
            st.error("Not enough data available for the selected duration. Try a longer duration.")
        else:
            st.success("Data loaded. Training model...")

            window_size = 180
            X_scaled, y_scaled, scaler = preprocess_data(data, window_size)

            model = build_model((window_size, 1))
            model.fit(X_scaled, y_scaled, batch_size=32, epochs=40, verbose=0)

            future_predictions = predict_future(model, X_scaled[-1:], predict_days, scaler)
            future_dates = pd.date_range(start=end_date + pd.Timedelta(days=1), periods=predict_days, freq='B')

            df_future = pd.DataFrame({'Date': future_dates, 'Predicted Close': future_predictions})

            st.subheader(f"📊 Predicted Closing Prices for Next {predict_days} Days")
            st.line_chart(df_future.set_index('Date'))
            st.dataframe(df_future)

    except Exception as e:
        st.error(f"Error: {e}")
