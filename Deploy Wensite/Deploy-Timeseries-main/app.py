import datetime
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib

matplotlib.use('agg')

app = Flask(__name__)
model = load_model('news_test_latest-2.keras')

# Function to calculate sine and cosine of the day of the year
def encode_date_as_features(date_series):
    day_of_year = date_series.dt.dayofyear
    sin_feature = np.sin(2 * np.pi * day_of_year / 365)
    cos_feature = np.cos(2 * np.pi * day_of_year / 365)
    return sin_feature, cos_feature

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get the number of future days to predict
        future_days = int(request.form['future_days'])

        # Read data from CSV file
        df = pd.read_csv("data_TA.csv", delimiter=";")

        # Interpolate and fill missing values
        df.interpolate(method='linear', inplace=True)
        df['Tx'].fillna(method='bfill', inplace=True)

        # Convert 'Tanggal' to datetime format and encode dates as features
        df['Tanggal'] = pd.to_datetime(df['Tanggal'], format='%d/%m/%Y', dayfirst=True)
        sin_feature, cos_feature = encode_date_as_features(df['Tanggal'])
        temperature = df['Tx'].astype(float)

        # Normalize temperature data
        scaler = MinMaxScaler(feature_range=(0, 1))
        temperature_scaled = scaler.fit_transform(temperature.values.reshape(-1, 1))

        # Combine temperature and date features for the last 'time_steps' from the training data
        last_sequence = np.hstack((
            temperature_scaled[-10:],  # Last 10 temperature values
            sin_feature.values[-10:].reshape(-1, 1),  # Last 10 sine values
            cos_feature.values[-10:].reshape(-1, 1)   # Last 10 cosine values
        ))

        # Reshape the last_sequence to match the input shape (1, time_steps, n_cols)
        last_sequence = np.reshape(last_sequence, (1, 10, 3))

        # List to store the predicted future values
        future_predictions = []

        # Forecast the next 'future_days' days
        for i in range(future_days):
            # Predict the next value
            next_pred = model.predict(last_sequence)

            # Append the predicted value to the list
            future_predictions.append(next_pred[0, 0])

            # Prepare the next sequence by shifting and adding new prediction
            next_pred_reshaped = np.reshape(next_pred, (1, 1, 1))  # Reshape to match (1, 1, 1)
            
            # Generate the sine and cosine values for the next day
            next_day = (df['Tanggal'].dt.dayofyear.iloc[-1] + i + 1) % 365
            next_sin = np.sin(2 * np.pi * next_day / 365).reshape(1, 1, 1)
            next_cos = np.cos(2 * np.pi * next_day / 365).reshape(1, 1, 1)
            
            # Concatenate the prediction with sine and cosine values
            next_sequence = np.concatenate((next_pred_reshaped, next_sin, next_cos), axis=2)
            
            # Update the last_sequence by removing the first element and adding the new sequence at the end
            last_sequence = np.append(last_sequence[:, 1:, :], next_sequence, axis=1)

        # Convert predictions to a numpy array
        future_predictions = np.array(future_predictions)
        forecast_original = scaler.inverse_transform(future_predictions.reshape(-1, 1))

        # Plot only prediction data
        plt.figure(figsize=(15, 10))
        plt.plot(np.arange(1, future_days + 1), forecast_original,
                 label='Forecast', linestyle='dashed')
        plt.scatter(np.arange(1, future_days + 1), forecast_original, color='red')

        for i, txt in enumerate(forecast_original):
            plt.annotate(f'{txt[0]:.2f}', (i + 1, forecast_original[i]),
                         textcoords="offset points", xytext=(0, 5), ha='center')

        plt.xlabel('Days into the Future')
        plt.ylabel('Temperature')
        plt.title('Future Temperature Forecast')
        plt.legend()

        # Save the plot for the forecast
        now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        image_path = f"static/{now}.png"
        plt.savefig(image_path)

        # Plot the original data and future predictions
        plt.figure(figsize=(14, 5))

        # Plot the original training data
        plt.plot(np.arange(len(temperature_scaled)), temperature_scaled, color='blue', label='Original Data')

        # Plot the future predictions
        plt.plot(np.arange(len(temperature_scaled), len(temperature_scaled) + future_days), future_predictions, color='red', label='Future Predictions')

        # Add labels and title
        plt.title('Future Forecasting')
        plt.xlabel('Time Steps')
        plt.ylabel('Tx Value')
        plt.legend()

        # Show the plot (optional in the Flask app)
        # plt.show()

        # Save the combined plot of original and future data
        combined_image_path = f"static/{now}_combined.png"
        plt.savefig(combined_image_path)

        return render_template("pages/result.html",
                               forecast=forecast_original[-1][0],
                               future_days=future_days,
                               image_path=image_path,
                               combined_image_path=combined_image_path)
    else:
        return render_template("pages/index.html")


if __name__ == "__main__":
    app.run(debug=True, port=8080)
