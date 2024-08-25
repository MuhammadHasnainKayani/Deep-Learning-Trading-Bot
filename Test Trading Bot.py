import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from binance.client import Client
import joblib  # For loading the saved scaler


scaler = joblib.load('scaler_updated.joblib')

# Load the saved model
model = load_model('Trading_bot_Bitcoin_BiGRU_updated.keras')  # Updated file format

# Binance API keys (Do not hardcode these in practice)
api_key = 'api_key'
api_secret = 'secret_key'
client = Client(api_key, api_secret)

# Fetch current klines data
symbol = 'BTCUSDT'
interval = Client.KLINE_INTERVAL_5MINUTE  # Adjust this to match your training interval
klines = client.get_klines(symbol=symbol, interval=interval)

# Convert klines data to DataFrame
columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
data = pd.DataFrame(klines, columns=columns)

# Convert timestamp to datetime and set it as the index
data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
data.set_index('timestamp', inplace=True)

# Drop 'ignore' column
data = data.drop(['ignore'], axis=1)



# Preprocess the data
sequence_length = 60
target_length = 10
num_features = data.shape[1]

# Normalize the last sequence_length data for prediction
# Calculate the middle index
middle_index = len(data) // 2

input_sequence = data[middle_index:middle_index+60]
scaled_data = scaler.transform(input_sequence)

# Prepare the input for prediction
normalized_sequence = scaled_data.reshape(1, sequence_length, num_features)

# Get prediction from the model
prediction = model.predict(normalized_sequence)

# Reshape and inverse transform to get original scale
predicted_values = prediction.reshape(target_length, num_features)
predicted_values = scaler.inverse_transform(predicted_values)

# Create a DataFrame for the predicted values
predicted_data = pd.DataFrame(predicted_values, columns=data.columns)
predicted_data.index = pd.date_range(start=data.index[-1] + pd.Timedelta(minutes=5), periods=target_length, freq='5T')

# Print the predicted values
print("Predicted Values for the Next 10 Time Steps:")
print(predicted_data)

# Combine original data and predicted data for plotting
combined_data = pd.concat([data.tail(sequence_length), predicted_data])

# Plot input sequence and prediction
plt.figure(figsize=(14, 7))

# Plot input sequence
plt.subplot(2, 1, 1)
plt.plot(data.index[-sequence_length:], input_sequence['close'], label='Input Sequence')
plt.title('Input Sequence')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()

# Plot prediction
plt.subplot(2, 1, 2)
plt.plot(predicted_data.index, predicted_data['close'], label='Prediction Output', color='orange')
plt.title('Prediction Output')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()

plt.tight_layout()
plt.show()

# Create candlestick chart
fig = go.Figure(data=[go.Candlestick(x=combined_data.index,
                open=combined_data['open'],
                high=combined_data['high'],
                low=combined_data['low'],
                close=combined_data['close'],
                increasing_line_color='blue',
                decreasing_line_color='red')])

# Update layout for better visualization
fig.update_layout(title='Trading Chart - Predicted vs. Actual',
                  xaxis_title='Time',
                  yaxis_title='Price',
                  xaxis_rangeslider_visible=False,
                  template='plotly_dark')  # You can change the template to 'plotly' for a light theme

# Save the figure to an HTML file
fig.write_html('trading_chart.html')

# Print a message indicating the file has been saved
print("Trading chart saved to 'trading_chart.html'")

# Show the figure
fig.show()
