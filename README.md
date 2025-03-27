# Stock Price Prediction with LSTM

This Python script employs **Long Short-Term Memory (LSTM)** neural networks to predict stock prices using historical data from Yahoo Finance. LSTMs excel at capturing temporal dependencies in sequential data, making them well-suited for time series forecasting.

## 🚀 Key Features

- **LSTM Model** – Utilizes a **Sequential LSTM** neural network to analyze historical stock prices and make predictions.  
- **Hyperparameter Tuning** – Integrated grid search optimizes key parameters like **epochs, batch size, and prediction days** for enhanced performance.  
- **Bootstrap Predictions** – Employs bootstrap resampling to quantify uncertainty and provide a range of potential outcomes for risk assessment.  
- **Real-time Prediction** – Generates **next-day stock price forecasts**, offering timely insights for decision-making.  
- **Visualization** – Automatically generates and saves charts comparing actual vs. predicted stock prices.  

## 📦 Installation

Ensure you have Python installed. Then, install the required dependencies:

```bash
pip install -r requirements.txt