import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
import joblib

# Define the model structure
model = Sequential([
    Dense(128, activation='relu', input_shape=(3,)),
    Dense(256, activation='relu'),
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')  # Output layer for regression
])

def parametricBodyFatEstimation(sex, height, neck, abdomen):
    model_path = None
    scaler_path = None

    if sex == 'male':
        model_path = '../models/men_best_model_weights.h5'
        scaler_path = '../scalers/men_scaler.joblib'
    elif sex == 'female':
        model_path = '../models/women_best_model_weights.h5'
        scaler_path = '../scalers/women_scaler.joblib'
    else:
        return "Invalid sex"

    model.load_weights(model_path)
    scaler = joblib.load(scaler_path)

    # Prepare the input
    X = pd.DataFrame([[height, neck, abdomen]], columns=['Height', 'Neck', 'Abdomen'])
    X = scaler.transform(X)

    # Make prediction
    predicted_value = model.predict(X, verbose=0)

    return predicted_value[0][0]