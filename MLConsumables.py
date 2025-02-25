import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import pandas as pd
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pymongo import MongoClient
from datetime import datetime, timedelta
from dotenv import load_dotenv
import logging
import time
from tensorflow.keras.callbacks import Callback

# Custom callback to display current epoch
class EpochLogger(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print(f"\nEpoch {epoch+1}/{self.params['epochs']} starting...")
    
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch+1}/{self.params['epochs']} completed. Loss: {logs['loss']:.4f}, Val Loss: {logs['val_loss']:.4f}")


def connect_to_mongodb():
    """
    Establish connection to MongoDB using the same structure as the GUI
    """
    try:
        client = MongoClient(os.getenv("MONGODB_URI"))
        db = client["LunarVitalsDB"]
        collection = db["sensor_data"]
        print("Successfully connected to MongoDB")
        return collection
    except Exception as e:
        logging.error(f"Error connecting to MongoDB: {e}")
        return None

def fetch_latest_data():
    """
    Fetch the most recent data point to check structure
    """
    collection = connect_to_mongodb()
    if collection is not None:
        try:
            latest = collection.find_one(sort=[('timestamp', -1)])
            if latest is not None:
                print("\nMost recent document structure:")
                print(latest)
            return latest
        except Exception as e:
            print(f"Error fetching latest data: {e}")
            return None

def fetch_sensor_data():
    """
    Fetch sensor data from MongoDB with optional time range filter
    """
    collection = connect_to_mongodb()
    if collection is None:
        return None
    
    query = {}

    try:
        # First check what activities are available
        activities = collection.distinct('activity_id', query)
        print(f"\nAvailable activities in time range: {activities}")

        # Get document count
        doc_count = collection.count_documents(query)
        print(f"Documents in time range: {doc_count}")
        
        # max_docs = 10000
        # print(f"Limiting to {max_docs} documents for testing")
        # cursor = collection.find(query).sort('timestamp', 1).limit(max_docs)

        # Fetch the actual data
        cursor = collection.find(query).sort('timestamp', 1)
        data = list(cursor)
        
        if data:
            print(f"\nSuccessfully retrieved {len(data)} documents")
        
        return data

    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def process_sensor_data(data):
    """
    Process and combine sensor data into a single DataFrame
    """
    if not data:
        print("No data to process")
        return None
        
    # Initialize empty lists to store the processed data
    processed_data = []
    
    # Group documents by timestamp and activity_id
    timestamps = set(d['timestamp'] for d in data if 'timestamp' in d)
    
    for timestamp in timestamps:
        docs_at_timestamp = [d for d in data if d.get('timestamp') == timestamp]
        if not docs_at_timestamp:
            continue
            
        record = {
            'timestamp': timestamp,
            'activity_id': docs_at_timestamp[0].get('activity_id', 'unknown')
        }
        
        # Extract relevant features
        for doc in docs_at_timestamp:
            if 'PulseSensor' in doc:
                record['pulse_BPM'] = doc['PulseSensor'].get('pulse_BPM', 0)
            if 'RespiratoryRate' in doc:
                record['BRPM'] = doc['RespiratoryRate'].get('BRPM', 0)
            if 'MPU_Gyroscope' in doc:
                record['steps'] = doc['MPU_Gyroscope'].get('steps', 0)
            if 'MLX_ObjectTemperature' in doc:
                record['body_temp'] = doc['MLX_ObjectTemperature'].get('Celsius', 0)
        
        # Only add records that have all required fields
        if all(k in record for k in ['pulse_BPM', 'BRPM', 'steps', 'body_temp']):
            processed_data.append(record)
    
    df = pd.DataFrame(processed_data)
    print(f"Processed {len(df)} complete records")
    return df

def calculate_air_consumption(row):
    """
    Calculate estimated oxygen consumption based on biometric data
    Uses accurate physiological models based on astronaut data:
    - 1600L total oxygen supply (2x 800L tanks)
    - Base consumption of ~50L/hour
    - BRPM: ~1.05L oxygen per 30 BRPM
    - HR: Variable consumption based on heart rate
    - Temperature: 5% increase per °C above normal (37°C)
    - Steps: 0.01L per step
    
    Returns oxygen consumption in liters per minute
    """
    # 1. Calculate respiratory oxygen consumption (based on breathing rate)
    # Linear interpolation for respiratory oxygen consumption
    if row['BRPM'] <= 15:
        respiratory_consumption = 0.45
    elif row['BRPM'] >= 30:
        respiratory_consumption = 1.05
    else:
        # Linear interpolation between known points
        respiratory_consumption = 0.45 + ((row['BRPM'] - 15) / (30 - 15)) * (1.05 - 0.45)
    
    # 2. Calculate heart rate based oxygen consumption
    # Map heart rate to oxygen consumption (L/min)
    if row['pulse_BPM'] <= 60:
        hr_consumption = 0.25  # At rest
    elif row['pulse_BPM'] <= 100:
        # Linear interpolation between rest (60bpm, 0.25L/min) and light exercise (100bpm, 1.0L/min)
        hr_factor = (row['pulse_BPM'] - 60) / (100 - 60)
        hr_consumption = 0.25 + hr_factor * (1.0 - 0.25)
    elif row['pulse_BPM'] <= 140:
        # Linear interpolation between light (100bpm, 1.0L/min) and moderate exercise (140bpm, 2.0L/min)
        hr_factor = (row['pulse_BPM'] - 100) / (140 - 100)
        hr_consumption = 1.0 + hr_factor * (2.0 - 1.0)
    elif row['pulse_BPM'] <= 180:
        # Linear interpolation between moderate (140bpm, 2.0L/min) and heavy exercise (180bpm, 3.5L/min)
        hr_factor = (row['pulse_BPM'] - 140) / (180 - 140)
        hr_consumption = 2.0 + hr_factor * (3.5 - 2.0)
    else:
        # Linear interpolation between heavy (180bpm, 3.5L/min) and maximal (200bpm, 5.0L/min)
        hr_factor = min(1.0, (row['pulse_BPM'] - 180) / (200 - 180))
        hr_consumption = 3.5 + hr_factor * (5.0 - 3.5)
    
    # 3. Temperature effect on metabolic rate
    # Normal body temperature is 37°C, metabolic rate increases ~5% per °C
    temp_difference = row['body_temp'] - 37.0
    if temp_difference > 0:
        # Increase metabolic rate by 5% per degree above normal
        temp_factor = 1.0 + (0.05 * temp_difference)
    else:
        # For temperatures below normal, we'll assume a smaller effect
        temp_factor = 1.0 + (0.02 * temp_difference)
    
    # 4. Physical activity (steps)
    # We need to convert steps per minute to a consumption rate
    # The 'steps' value from sensor might be cumulative, so we'll use it as a proxy for activity level
    # Assuming 'steps' represents steps per minute or recent activity level
    step_consumption = row['steps'] * 0.014
    
    # 5. Combine the factors
    # We'll use the max of respiratory and heart rate consumption as they overlap
    # Then add effects of temperature and physical activity
    base_consumption = max(respiratory_consumption, hr_consumption)
    total_consumption = (base_consumption * temp_factor) + step_consumption
    
    # Ensure we don't go below a minimum reasonable value (0.2 L/min)
    total_consumption = max(0.2, total_consumption)
    
    return total_consumption

def create_training_data(df):
    """
    Create training data with features and target variable
    """
    if df is None or df.empty:
        print("No data available for training")
        return None, None, None, None
        
    # Calculate air consumption for each record
    df['air_consumption'] = df.apply(calculate_air_consumption, axis=1)
    
    # Features for the model
    X = df[['pulse_BPM', 'BRPM', 'steps', 'body_temp']]
    y = df['air_consumption']
    
    # Convert consumption to liters per hour for easier interpretation
    print(f"Average oxygen consumption: {y.mean():.2f} L/min ({y.mean()*60:.1f} L/hour)")
    print(f"Min oxygen consumption: {y.min():.2f} L/min")
    print(f"Max oxygen consumption: {y.max():.2f} L/min")
    
    # Calculate remaining time with 1600L of oxygen
    total_oxygen = 1600  # 2 tanks of 800L each
    avg_consumption_per_hour = y.mean() * 60
    remaining_hours = total_oxygen / avg_consumption_per_hour
    print(f"Estimated remaining time with 1600L oxygen supply: {remaining_hours:.1f} hours ({remaining_hours/24:.1f} days)")
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

def build_model(input_shape):
    """
    Create and compile the neural network model
    """
    model = Sequential([
        Dense(64, activation='relu', input_shape=input_shape),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_and_evaluate(data):
    """
    Main function to process data, train model, and evaluate results
    """
    # Process the raw sensor data
    df = process_sensor_data(data)
    if df is None:
        return None, None, None
    
    # Create training data
    X_train, X_test, y_train, y_test = create_training_data(df)
    if X_train is None:
        return None, None, None
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Build and train the model
    model = build_model((X_train.shape[1],))
    
    # Create the epoch logger callback
    epoch_logger = EpochLogger()
    
    # Train the model with the custom callback
    print("\n--- Starting model training ---")
    history = model.fit(
        X_train_scaled, y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        verbose=1,
        callbacks=[epoch_logger]
    )
    print("--- Model training complete ---\n")
    
    # Calculate hours of remaining oxygen based on predictions
    predictions = model.predict(X_test_scaled)
    avg_predicted_consumption = np.mean(predictions) * 60  # Convert to L/hour
    remaining_hours = 1600 / avg_predicted_consumption
    print(f"\nAverage predicted oxygen consumption: {avg_predicted_consumption:.2f} L/hour")
    print(f"Estimated remaining time with 1600L supply: {remaining_hours:.1f} hours ({remaining_hours/24:.1f} days)")
    
    return model, scaler, history

def predict_remaining_oxygen(model, scaler, new_data):
    """
    Predict oxygen consumption and calculate remaining supply time
    """
    df_new = process_sensor_data(new_data)
    if df_new is None or df_new.empty:
        return None
    
    # Extract features
    features = df_new[['pulse_BPM', 'BRPM', 'steps', 'body_temp']]
    
    # Scale features
    scaled_features = scaler.transform(features)
    
    # Predict oxygen consumption (L/min)
    predictions = model.predict(scaled_features)
    
    # Calculate remaining oxygen
    total_oxygen = 1600  # 2 tanks of 800L each
    df_new['predicted_consumption_L_min'] = predictions
    df_new['predicted_consumption_L_hour'] = predictions * 60
    
    # Calculate average consumption
    avg_consumption = np.mean(predictions) * 60  # L/hour
    remaining_hours = total_oxygen / avg_consumption
    
    print(f"\nOxygen Supply Estimates:")
    print(f"Average predicted consumption: {avg_consumption:.2f} L/hour")
    print(f"Estimated remaining time: {remaining_hours:.1f} hours ({remaining_hours/24:.1f} days)")
    
    return df_new

def main():    
    # Fetch data from MongoDB
    print("Fetching data from MongoDB...")
    raw_data = fetch_sensor_data()
    
    if raw_data:
        print("Training model...")
        model, scaler, history = train_and_evaluate(raw_data)
        
        if model and scaler:
            print("\nPredicting oxygen consumption for new data...")
            new_data = fetch_sensor_data()  # Normally would be new data
            
            if new_data:
                result_df = predict_remaining_oxygen(model, scaler, new_data)
                if result_df is not None:
                    print("\nSample predictions:")
                    print(result_df['predicted_consumption_L_min'].head())

if __name__ == "__main__":
    main()