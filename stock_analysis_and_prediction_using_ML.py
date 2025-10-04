import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# --- 1. Data Collection ---
# Download historical data (Example: Apple stock 'AAPL')
ticker = 'JIOFIN.NS'
start_date = '2010-01-01'
end_date = '2025-10-01'

# The yf.download() function successfully retrieves the data
print(f"Downloading historical data for {ticker}...")
df = yf.download(ticker, start=start_date, end=end_date)
print("Download complete.")

# FIX: Use double brackets [['Close']] to ensure the data is a 2D DataFrame (n_samples, 1)
# This prevents the MinMaxScaler error.
data = df[['Close']]
dataset = data.values

# Optional: Print the shape to confirm the fix
print(f"Shape of dataset before scaling: {dataset.shape}")

# --- 2. Data Preprocessing ---
# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
# The fit_transform now works because dataset has shape (3960, 1)
scaled_data = scaler.fit_transform(dataset)

# Define the training data size (e.g., 80% of the data)
training_data_len = int(np.ceil(len(scaled_data) * 0.8))

# Create the training data set
train_data = scaled_data[0:training_data_len, :]

# Define time step (e.g., look back 60 days to predict the next day)
time_step = 60
X_train = []
y_train = []

for i in range(time_step, len(train_data)):
    X_train.append(train_data[i-time_step:i, 0])
    y_train.append(train_data[i, 0])

# Convert X_train and y_train to numpy arrays and reshape for LSTM
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# --- 3. Model Building (LSTM) ---
print("\nBuilding and training LSTM model...")
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2)) 
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# --- 4. Training ---
# Train the model (adjust epochs and batch_size as needed)
model.fit(X_train, y_train, batch_size=1, epochs=1, verbose=0)
print("Training complete.")

# --- 5. Prediction Setup ---
# Create the testing data set
test_data = scaled_data[training_data_len - time_step:, :]
X_test = []
y_test = dataset[training_data_len:, :] # Actual unscaled prices

for i in range(time_step, len(test_data)):
    X_test.append(test_data[i-time_step:i, 0])

# Convert to numpy arrays and reshape for LSTM
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Get the model's predicted price values
predictions = model.predict(X_test)
# Inverse transform the predictions to get actual price values
predictions = scaler.inverse_transform(predictions)

# --- 6. Visualization and Evaluation ---
# Create a DataFrame for plotting
train = data[:training_data_len]
valid = data[training_data_len:].copy() # Use .copy() to avoid FutureWarning
valid['Predictions'] = predictions

# Print results
print("\n--- Model Evaluation ---")
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(valid['Close'], valid['Predictions']))
print(f"Root Mean Square Error (RMSE): {rmse:.2f}")

print("\nLast 5 Actual Prices:")
print(valid['Close'].tail())
print("\nLast 5 Predicted Prices:")
print(valid['Predictions'].tail())

# Plot the data
plt.figure(figsize=(16, 8))
plt.title(f'Stock Price Prediction for {ticker}')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'], label='Train')
plt.plot(valid['Close'], label='Actual Value')
plt.plot(valid['Predictions'], label='Predictions')
plt.legend(loc='lower right')
plt.show()