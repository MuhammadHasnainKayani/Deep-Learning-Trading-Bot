import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.regularizers import l2
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load data
data = pd.read_csv('1_year_Bitcoin_historical_data(5-min).csv')
data['timestamp'] = pd.to_datetime(data['timestamp'])
data.set_index('timestamp', inplace=True)

# Drop 'ignore' column
data = data.drop(['ignore'], axis=1)

# Function to create sequences for training
def create_sequences(data, sequence_length, target_length):
    X, y = [], []
    for i in range(len(data) - sequence_length - target_length + 1):
        seq = data.iloc[i:i + sequence_length].values
        label = data.iloc[i + sequence_length:i + sequence_length + target_length].values
        X.append(seq)
        y.append(label)
    return np.array(X), np.array(y)

# Function to normalize data
def normalize_data(data, scaler=None):
    if not scaler:
        scaler = MinMaxScaler()
        scaler.fit(data)
    scaled_data = scaler.transform(data)
    return scaled_data, scaler

# Define the BiGRU model
def build_bigru_model(input_shape, target_length, num_features, dropout_rate=0.3, l2_value=0.01):
    model = Sequential([
        Bidirectional(GRU(128, return_sequences=True, kernel_regularizer=l2(l2_value)), input_shape=input_shape),
        BatchNormalization(),
        Dropout(dropout_rate),
        Bidirectional(GRU(128, return_sequences=True, kernel_regularizer=l2(l2_value))),
        BatchNormalization(),
        Dropout(dropout_rate),
        Bidirectional(GRU(128, kernel_regularizer=l2(l2_value))),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(target_length * num_features, kernel_regularizer=l2(l2_value))
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

# Prepare data for training
sequence_length = 60
target_length = 10
num_features = data.shape[1]

input_data, scaler = normalize_data(data)
X, y = create_sequences(pd.DataFrame(input_data, columns=data.columns), sequence_length, target_length)

# Flatten the target data
y = y.reshape(-1, target_length * num_features)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape data for BiGRU input
X_train = X_train.reshape(X_train.shape[0], sequence_length, X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], sequence_length, X_test.shape[2])

# Build and train the BiGRU model
model = build_bigru_model((X_train.shape[1], X_train.shape[2]), target_length, num_features)
history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test), verbose=1)

# Save the model
model.save('Trading_bot_Bitcoin_BiGRU.keras')

# Evaluate the model on the test set
y_pred = model.predict(X_test)

# Reshape the predictions back to the original shape
y_pred_inverse = y_pred.reshape(-1, target_length, num_features)
y_test_inverse = y_test.reshape(-1, target_length, num_features)

# Evaluate the model performance
mse = mean_squared_error(y_test_inverse.flatten(), y_pred_inverse.flatten())
mae = mean_absolute_error(y_test_inverse.flatten(), y_pred_inverse.flatten())

print(f'Mean Squared Error on Test Set: {mse}')
print(f'Mean Absolute Error on Test Set: {mae}')
