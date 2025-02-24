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

        if doc_count == 0:
            # Check total documents and time range in collection
            total_docs = collection.count_documents({})
            print(f"\nTotal documents in collection: {total_docs}")
            
            time_range = list(collection.aggregate([
                {
                    '$group': {
                        '_id': None,
                        'min_time': {'$min': '$timestamp'},
                        'max_time': {'$max': '$timestamp'}
                    }
                }
            ]))
            
            if time_range:
                min_time = datetime.fromtimestamp(time_range[0]['min_time'])
                max_time = datetime.fromtimestamp(time_range[0]['max_time'])
                print(f"\nAvailable data range:")
                print(f"From: {min_time}")
                print(f"To: {max_time}")

        # Fetch the actual data
        cursor = collection.find(query).sort('timestamp', 1)
        data = list(cursor)
        
        if data:
            print(f"\nSuccessfully retrieved {len(data)} documents")
            print("Sample data fields available:")
            for key in data[0].keys():
                print(f"- {key}")
        
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
    Calculate estimated air consumption based on biometric data
    This is a simplified model - you'll need to adjust the coefficients based on real data
    """
    # Base consumption rate (liters per minute)
    base_rate = 0.5
    
    # Adjustments based on biometric data
    hr_factor = 0.01 * row['pulse_BPM']
    rr_factor = 0.05 * row['BRPM']
    step_factor = 0.002 * row['steps']
    temp_factor = 0.05 * (row['body_temp'] - 37.0)
    
    return base_rate * (1 + hr_factor + rr_factor + step_factor + temp_factor)

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
    
    # Train the model
    history = model.fit(
        X_train_scaled, y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        verbose=1
    )
    
    # Evaluate the model
    test_results = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Test Loss: {test_results[0]:.4f}")
    print(f"Test MAE: {test_results[1]:.4f}")
    
    return model, scaler, history

def main():    
    # Fetch data from MongoDB
    print("Fetching data from MongoDB...")
    raw_data = fetch_sensor_data()
    
    if raw_data:
        print("Training model...")
        model, scaler, history = train_and_evaluate(raw_data)
        
        if model and scaler:
            # Example: Make predictions for new data
            new_data = fetch_sensor_data((datetime.now() + timedelta(minutes=5)).timestamp()
            )
            
            if new_data:
                df_new = process_sensor_data(new_data)
                if df_new is not None and not df_new.empty:
                    new_features = df_new[['pulse_BPM', 'BRPM', 'steps', 'body_temp']]
                    predictions = model.predict(scaler.transform(new_features))
                    
                    # Add predictions to DataFrame
                    df_new['predicted_air_consumption'] = predictions
                    print("\nPredictions:")
                    print(df_new[['timestamp', 'activity_id', 'predicted_air_consumption']])

if __name__ == "__main__":
    main()