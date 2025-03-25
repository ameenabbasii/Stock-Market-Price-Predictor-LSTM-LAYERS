import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display


def preprocess_data(data, prediction_days):
    if data.empty:
        raise ValueError("Empty dataset. Please check the data.")
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    x_train, y_train = [], []
    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x - prediction_days: x, 0])
        y_train.append(scaled_data[x, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    return x_train, y_train, scaler


def create_and_train_model(x_train, y_train, epochs, batch_size, loading_bar):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    loading_bar.max = epochs

    for epoch in range(epochs):
        model.fit(x_train, y_train, epochs=1, batch_size=batch_size, verbose=0)
        loading_bar.value = epoch + 1

    return model


def predict_stock_prices(model, data, scaler, prediction_days):
    model_inputs = data['Close'].values.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)

    x_test = []
    for x in range(prediction_days, len(model_inputs)):
        x_test.append(model_inputs[x - prediction_days:x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    return predicted_prices


def save_plot_as_image(actual_prices, predicted_prices, company):
    plt.plot(actual_prices, color="black", label=f"Actual {company} Prices")
    plt.plot(predicted_prices, color="green", label=f"Predicted {company} Prices")
    plt.title(f"{company} Share Prices")
    plt.xlabel('Time')
    plt.ylabel(f"{company} Share Price")
    plt.legend()


    plt.savefig(f"{company}_stock_prediction.png")
    plt.close() 


def plot_stock_prices(actual_prices, predicted_prices, company):
    plt.plot(actual_prices, color="black", label=f"Actual {company} Prices")
    plt.plot(predicted_prices, color="green", label=f"Predicted {company} Prices")
    plt.title(f"{company} Share Prices")
    plt.xlabel('Time')
    plt.ylabel(f"{company} Share Price")
    plt.legend()


    display(plt.gcf())


def process_company(widget):
    company = entry_company.value
    prediction_days = entry_prediction_days.value
    epochs = entry_epochs.value
    batch_size = entry_batch_size.value
    start_date = '2010-01-01'
    end_date_training = '2022-01-01'
    end_date_testing = dt.datetime.now().strftime('%Y-%m-%d')

    try:

        data_training = yf.download(company, start=start_date, end=end_date_training)
        x_train, y_train, scaler = preprocess_data(data_training, prediction_days)


        loading_bar = widgets.IntProgress(
            value=0,
            min=0,
            max=epochs,
            description='Training:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='50%')
        )

        display(loading_bar)


        model = create_and_train_model(x_train, y_train, epochs, batch_size, loading_bar)


        data_testing = yf.download(company, start=end_date_training, end=end_date_testing)

        if data_testing.empty:
            raise ValueError("Empty testing dataset. Please check the data.")

        actual_prices = data_testing['Close'].values
        predicted_prices = predict_stock_prices(model, data_testing, scaler, prediction_days)

        plot_stock_prices(actual_prices, predicted_prices, company)

        save_plot_as_image(actual_prices, predicted_prices, company)

        last_days_data = data_testing['Close'].values[-prediction_days:].reshape(-1, 1)
        last_days_data = scaler.transform(last_days_data)
        real_data = np.reshape(last_days_data, (1, prediction_days, 1))
        prediction_real_time = model.predict(real_data)
        prediction_real_time = scaler.inverse_transform(prediction_real_time)

        result_label.value = f"Real-time Prediction for Tomorrow: {prediction_real_time[0][0]}"

    except Exception as e:
        result_label.value = f"Error: {e}"


entry_company = widgets.Text(description="Enter Company:")
entry_prediction_days = widgets.IntText(description="Enter Prediction Days:")
entry_epochs = widgets.IntText(description="Enter Epochs:")
entry_batch_size = widgets.IntText(description="Enter Batch Size:")
btn_process = widgets.Button(description="Process Company")
result_label = widgets.Label()

btn_process.on_click(process_company)

# Display widgets
display(entry_company)
display(entry_prediction_days)
display(entry_epochs)
display(entry_batch_size)
display(btn_process)
display(result_label)


