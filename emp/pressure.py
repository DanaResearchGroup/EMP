import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Load your dataset (replace 'your_data.csv' with your actual dataset)
data = pd.read_csv('data.csv')

# Assume your dataset has columns 'feature1', 'feature2', ..., 'detonation_pressure'
# Adjust the column names accordingly
features = data.drop('detonation_pressure', axis=1).values
labels = data['detonation_pressure'].values

# Split the data into training and testing sets
features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.2, random_state=42
)

# Standardize features
scaler = StandardScaler()
features_train = scaler.fit_transform(features_train)
features_test = scaler.transform(features_test)

# Build the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(features_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)  # Output layer with 1 neuron for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(features_train, labels_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model
labels_pred = model.predict(features_test)
mse = mean_squared_error(labels_test, labels_pred)
print(f'Mean Squared Error on Test Set: {mse}')
