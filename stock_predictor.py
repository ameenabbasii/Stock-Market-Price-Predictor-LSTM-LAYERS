import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from sklearn.metrics import mean_squared_error
from itertools import product
import ipywidgets as widgets
from IPython.display import display, HTML
import matplotlib.pyplot as plt

# Function to preprocess and scale the data
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

# Function to create and train the LSTM model
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

    # Calculate the total number of batches
    total_batches = int(np.ceil(x_train.shape[0] / batch_size))

    for epoch in range(epochs):
        for batch in range(total_batches):
            # Your training logic here

            # Update loading bar
            loading_bar.value += 1

    return model

# Function to predict stock prices
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

# Function to save the plot as an image
def save_plot_as_image(actual_prices, predicted_prices, company):
    plt.plot(actual_prices, color="black", label=f"Actual {company} Prices")
    plt.plot(predicted_prices, color="green", label=f"Predicted {company} Prices")
    plt.title(f"{company} Share Prices")
    plt.xlabel('Time')
    plt.ylabel(f"{company} Share Price")
    plt.legend()

    # Save the plot as an image file
    plt.savefig(f"{company}_stock_prediction.png")
    plt.close()  # Close the plot to avoid displaying it interactively

# Function to visualize actual and predicted prices
def plot_stock_prices(actual_prices, predicted_prices, company):
    plt.plot(actual_prices, color="black", label=f"Actual {company} Prices")
    plt.plot(predicted_prices, color="green", label=f"Predicted {company} Prices")
    plt.title(f"{company} Share Prices")
    plt.xlabel('Time')
    plt.ylabel(f"{company} Share Price")
    plt.legend()

    # Display the plot using the IPython display function
    display(plt.gcf())

# Function for hyperparameter tuning
def hyperparameter_tuning(data, company, end_date_training, end_date_testing):
    # Set up hyperparameter grid for grid search
    hyperparameter_grid = {
        'epochs': [25, 50, 100],
        'batch_size': [8, 16, 32],
        'prediction_days': [7, 14, 21]
    }

    best_model = None
    best_mse = float('inf')

    # Calculate the total number of combinations for loading bar
    total_combinations = len(list(product(*hyperparameter_grid.values())))
    loading_bar = widgets.IntProgress(
        value=0,
        min=0,
        max=total_combinations,
        description='Hyperparameter Tuning:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='50%')
    )
    display(loading_bar)

    for i, params in enumerate(product(*hyperparameter_grid.values())):
        epochs, batch_size, prediction_days = params

        x_train, y_train, scaler = preprocess_data(data, prediction_days)
        model = create_and_train_model(x_train, y_train, epochs, batch_size, loading_bar)

        # Download and preprocess testing data
        data_testing = yf.download(company, start=end_date_training, end=end_date_testing)

        if data_testing.empty:
            raise ValueError("Empty testing dataset. Please check the data.")

        # Predict stock prices
        actual_prices = data_testing['Close'].values
        predicted_prices = predict_stock_prices(model, data_testing, scaler, prediction_days)

        # Calculate Mean Squared Error (MSE)
        mse = mean_squared_error(actual_prices[prediction_days:], predicted_prices)

        if mse < best_mse:
            best_mse = mse
            best_model = model

        # Update loading bar
        loading_bar.value += 1

    return best_model, scaler, prediction_days, data_testing, company

# Function to bootstrap predictions for uncertainty estimation
def bootstrap_predictions(model, x_test, scaler, num_iterations=100):
    predicted_prices_distribution = []

    for _ in range(num_iterations):
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)
        predicted_prices_distribution.append(predictions.flatten())

    return np.array(predicted_prices_distribution)

# Function to fetch data, train model, and plot graph
def process_company(widget):
    company = entry_company.value
    start_date = '2010-01-01'
    end_date_training = '2022-01-01'
    end_date_testing = dt.datetime.now().strftime('%Y-%m-%d')

    try:
        # Download and preprocess training data
        data_training = yf.download(company, start=start_date, end=end_date_training)

        # Hyperparameter tuning
        best_model, scaler, prediction_days, data_testing, company = hyperparameter_tuning(
            data_training, company, end_date_training, end_date_testing
        )

        # Visualize actual and predicted prices using the best model
        actual_prices = data_testing['Close'].values
        x_test = data_testing['Close'].values[-prediction_days:]
        x_test = x_test.reshape(-1, 1)
        x_test = scaler.transform(x_test)
        x_test = np.reshape(x_test, (1, prediction_days, 1))

        # Bootstrap predictions for uncertainty estimation
        predicted_prices_distribution = bootstrap_predictions(best_model, x_test, scaler)

        # Plot actual and predicted prices
        plot_stock_prices(actual_prices, np.median(predicted_prices_distribution, axis=0), company)

        # Save the plot as an image
        save_plot_as_image(actual_prices, np.median(predicted_prices_distribution, axis=0), company)

        # Real-time prediction for tomorrow using the best model
        last_days_data = data_testing['Close'].values[-prediction_days:].reshape(-1, 1)
        last_days_data = scaler.transform(last_days_data)
        real_data = np.reshape(last_days_data, (1, prediction_days, 1))
        prediction_real_time = best_model.predict(real_data)
        prediction_real_time = scaler.inverse_transform(prediction_real_time)

        # Display the prediction in a nice font
        result_label.value = f"<p style='font-size:20px; color:blue;'>Best Model - Real-time Prediction for Tomorrow: {prediction_real_time[0][0]:.2f}</p>"

    except Exception as e:
        result_label.value = f"Error: {e}"

# GUI Components
entry_company = widgets.Text(description="Enter Company:")
btn_process = widgets.Button(description="Process Company")
result_label = widgets.HTML()

# Assign the callback function to the button
btn_process.on_click(process_company)

# Display widgets
display(entry_company)
display(btn_process)
display(result_label)
