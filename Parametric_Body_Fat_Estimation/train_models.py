import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
import joblib

# Load the dataset
file_path = 'Parametric_Body_Fat_Estimation/dataset/BodyFat - Extended.csv'
data = pd.read_csv(file_path)

data = data[data['Sex'] == 'F']

# Shuffle the dataset
data = data.sample(frac=1, random_state=42)

# Separate features and target variable
X = data[['Height', 'Neck', 'Abdomen']]
y = data["BodyFat"]

# Normalize the numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

joblib.dump(scaler, 'Parametric_Body_Fat_Estimation/scalers/women_scaler.joblib')

checkpoint_filepath = 'Parametric_Body_Fat_Estimation/models/women_best_model_weights.h5'
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_best_only=True,
    monitor='val_mean_absolute_error',
    mode='min',          # Save the model with minimum mean absolute error
    verbose=0
)

# Define the deep learning model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_scaled.shape[1],)),
    Dense(256, activation='relu'),
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')  # Output layer for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

print("Training women model...", end=' ', flush=True)
# Train the model
model.fit(
    X_scaled, y,
    validation_split=0.2,
    epochs=100,
    batch_size=16,
    callbacks=[checkpoint_callback],
    verbose=0
)
print("Done!")

# Load the dataset
data = pd.read_csv(file_path)

data = data[data['Sex'] == 'M']

# Shuffle the dataset
data = data.sample(frac=1, random_state=42)

# Separate features and target variable
X = data[['Height', 'Neck', 'Abdomen']]
y = data["BodyFat"]

# Normalize the numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

joblib.dump(scaler, 'Parametric_Body_Fat_Estimation/scalers/men_scaler.joblib')

checkpoint_filepath = 'Parametric_Body_Fat_Estimation/models/men_best_model_weights.h5'
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_best_only=True,
    monitor='val_mean_absolute_error',
    mode='min',          # Save the model with minimum mean absolute error
    verbose=0
)

# Define the deep learning model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_scaled.shape[1],)),
    Dense(256, activation='relu'),
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')  # Output layer for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

print("Training men model...", end=' ', flush=True)

# Train the model
history = model.fit(
    X_scaled, y,
    validation_split=0.2,
    epochs=100,
    batch_size=16,
    callbacks=[checkpoint_callback],
    verbose=0
)
print("Done!")