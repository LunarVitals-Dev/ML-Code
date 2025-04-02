import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import tensorflow as tf
from pymongo import MongoClient
from tensorflow.keras import layers
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
import zipfile
import joblib
from sklearn.metrics import classification_report, confusion_matrix

# Load environment variables for MongoDB connection
load_dotenv()

def extract_zip(zip_file_path, extract_to_folder):
    """Extracts zip files containing sensor data."""
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_folder)

def load_csv_data(extract_folder, file_name):
    """Loads CSV sensor data (e.g., heart rate, temperature)."""
    try:
        sampling_rate = pd.read_csv(os.path.join(extract_folder, file_name), header=None, nrows=1, skiprows=1, on_bad_lines='skip').iloc[0, 0]
        data = pd.read_csv(os.path.join(extract_folder, file_name), header=None, skiprows=2, on_bad_lines='skip')
        return data.values.astype(float), float(sampling_rate)
    except Exception as e:
        print(f"Error loading {file_name}: {e}")
        return None, None

def load_activity_data(folder_path, subject_id):
    """Loads activity timestamps and labels."""
    activity_file = os.path.join(folder_path, f"S{subject_id}", f"S{subject_id}_activity.csv")
    try:
        activity_df = pd.read_csv(activity_file, sep=',', header=None, skiprows=1, on_bad_lines='skip')
        activity_df.columns = ["activity_label", "start_time"]
        return activity_df
    except Exception as e:
        print(f"Error loading activity data for {activity_file}: {e}")
        return None

def create_activity_labels(activity_df, min_length, hr_rate):
    """Assigns activity labels based on timestamps."""
    activity_labels = np.full(min_length, 'NO_ACTIVITY', dtype='object')
    start_indices = (activity_df['start_time'] * hr_rate).astype(int)

    for i in range(len(activity_df)):
        start_index = start_indices[i]
        end_index = start_indices[i + 1] if i + 1 < len(activity_df) else min_length

        if start_index < min_length:
            activity_labels[start_index:min(end_index, min_length)] = activity_df['activity_label'].iloc[i]

    return activity_labels

def process_subject_data(folder_path, subject_id, extract_base_folder):
    """Processes heart rate, temperature, and activity data for a subject."""
    extract_folder = os.path.join(extract_base_folder, f"S{subject_id}")
    os.makedirs(extract_folder, exist_ok=True)

    zip_file_path = os.path.join(folder_path, f"S{subject_id}", f'S{subject_id}_E4.zip')
    if not os.path.exists(zip_file_path):
        print(f"Zip file not found: {zip_file_path}")
        return None

    extract_zip(zip_file_path, extract_folder)

    # Load sensor data - using only HR (BPM) and TEMP
    hr_values, hr_rate = load_csv_data(extract_folder, 'HR.csv')
    temp_values, temp_rate = load_csv_data(extract_folder, 'TEMP.csv')
    if hr_values is None or temp_values is None:
        return None

    # Downsample temperature data to match heart rate
    temp_downsample_factor = int(temp_rate / hr_rate)
    temp_values = temp_values[::temp_downsample_factor] if temp_downsample_factor > 0 else temp_values

    # Match lengths
    min_length = min(len(hr_values), len(temp_values))
    hr_values, temp_values = hr_values[:min_length], temp_values[:min_length]

    # Load activity labels
    activity_df = load_activity_data(folder_path, subject_id)
    if activity_df is None:
        return None

    activity_labels = create_activity_labels(activity_df, min_length, hr_rate)

    # Encode activity labels
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    activity_encoded = encoder.fit_transform(activity_labels.reshape(-1, 1))

    # Normalize features
    scaler = StandardScaler()
    hr_scaled = scaler.fit_transform(hr_values.reshape(-1, 1))
    temp_scaled = scaler.fit_transform(temp_values.reshape(-1, 1))

    X = np.hstack((hr_scaled, temp_scaled))  # Only using HR (BPM) and temperature
    Y = activity_encoded  # Classification target

    return X, Y, encoder, scaler

def train_activity_model():
    """Train the model on the PPG_FieldStudy dataset and save for later use."""
    folder_path = 'data/PPG_FieldStudy'
    extract_base_folder = 'extracted_files'
    all_subjects_X, all_subjects_Y = [], []
    encoder = None
    hr_scaler = StandardScaler()
    temp_scaler = StandardScaler()

    for subject_id in range(1, 16):
        subject_data = process_subject_data(folder_path, subject_id, extract_base_folder)
        if subject_data is not None:
            X, Y, encoder, _ = subject_data
            
            # Split X into HR and TEMP components (assuming it's a 2-column matrix)
            hr_data = X[:, 0].reshape(-1, 1)  # First column is HR
            temp_data = X[:, 1].reshape(-1, 1)  # Second column is TEMP
            
            # Fit scalers on the data
            hr_scaler.partial_fit(hr_data)
            temp_scaler.partial_fit(temp_data)
            
            all_subjects_X.append(X)
            all_subjects_Y.append(Y)
        else:
            print(f"Skipping subject S{subject_id}.")

    if not all_subjects_X:
        print("No data processed.")
        return None, None, None

    # Create a scaler dictionary
    scaler = {'bpm': hr_scaler, 'temp': temp_scaler}

    # Combine data from all subjects
    X_combined = np.concatenate(all_subjects_X, axis=0)
    Y_combined = np.concatenate(all_subjects_Y, axis=0)

    # Train-test split
    X_train, X_test, Y_train, Y_test = train_test_split(X_combined, Y_combined, test_size=0.2, random_state=42)

    num_classes = Y_train.shape[1]  # Number of unique activity classes

    # --- Neural Network Model ---
    # Note: Input shape is now 2 (BPM and body temperature) instead of more features
    model = tf.keras.Sequential([
        layers.InputLayer(shape=(2,)),  # Input shape is (2) for [BPM, body_temp]
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation='softmax')  # Classification output
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, Y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)

    # Evaluate model
    evaluation = model.evaluate(X_test, Y_test, verbose=1)
    print(f"Test Loss: {evaluation[0]:.4f}, Test Accuracy: {evaluation[1]:.4f}")

    # Predictions
    predictions = model.predict(X_test)
    predicted_labels = np.argmax(predictions, axis=1)
    actual_labels = np.argmax(Y_test, axis=1)

    # Convert numeric labels back to activity names
    decoded_predictions = encoder.inverse_transform(np.eye(num_classes)[predicted_labels])
    decoded_actuals = encoder.inverse_transform(np.eye(num_classes)[actual_labels])

    # Display some results
    for i in range(10):
        print(f"Predicted: {decoded_predictions[i][0]}, Actual: {decoded_actuals[i][0]}")

    # Print classification report
    activity_names = encoder.categories_[0]
    print("\nClassification Report:")
    print(classification_report(actual_labels, predicted_labels, target_names=activity_names))

    # Save model and preprocessing objects with correct format for Keras 3
    model.save('activity_model_bpm_temp.keras')  
    joblib.dump(encoder, 'activity_encoder_bpm_temp.joblib')
    joblib.dump({'bpm': hr_scaler, 'temp': temp_scaler}, 'feature_scaler_bpm_temp.joblib')

    print("Model, encoder, and scaler saved successfully.")
    return model, encoder, scaler

# --- MongoDB Connection & Data Fetching --- 
def connect_to_mongodb():
    """Connect to MongoDB and return the collection."""
    try:
        client = MongoClient(os.getenv("MONGODB_URI"))
        db = client["LunarVitalsDB"]
        collection = db["sensor_data"]
        return collection
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        return None

def fetch_sensor_data():
    """Fetch sensor data from MongoDB."""
    collection = connect_to_mongodb()
    if collection is None:
        return None
    
    print("Successfully connected to MongoDB")
    query = {}
    cursor = collection.find(query).sort('timestamp', 1)
    data = list(cursor)
    
    if data:
        print(f"\nSuccessfully retrieved {len(data)} documents")
    else:
        print("No data retrieved from MongoDB")
    
    return data

def process_sensor_data(data):
    """Process and combine sensor data into a DataFrame with averaged BPM."""
    if not data:
        return None
        
    processed_data = []
    
    timestamps = sorted(set(d['timestamp'] for d in data if 'timestamp' in d))
    
    for timestamp in timestamps:
        docs_at_timestamp = [d for d in data if d.get('timestamp') == timestamp]
        if not docs_at_timestamp:
            continue
            
        record = {
            'timestamp': timestamp,
            'activity_id': docs_at_timestamp[0].get('activity_id', 'unknown')
        }
        
        pulse_bpm = None
        respiratory_rate = None
        body_temp = None
        
        for doc in docs_at_timestamp:
            if 'PulseSensor' in doc:
                pulse_bpm = doc['PulseSensor'].get('pulse_BPM', None)
            if 'RespiratoryRate' in doc:
                respiratory_rate = doc['RespiratoryRate'].get('BRPM', None)
            if 'MLX_ObjectTemperature' in doc:
                body_temp = doc['MLX_ObjectTemperature'].get('Celsius', None)
        
        # Calculate average BPM if both values are available
        if pulse_bpm is not None and respiratory_rate is not None:
            record['avg_bpm'] = (pulse_bpm + respiratory_rate) / 2
        elif pulse_bpm is not None:
            record['avg_bpm'] = pulse_bpm
        elif respiratory_rate is not None:
            record['avg_bpm'] = respiratory_rate
        else:
            # Skip this record if no BPM data available
            continue
            
        if body_temp is not None:
            record['body_temp'] = body_temp
        else:
            # Skip this record if no temperature data available
            continue
        
        # Only keep records with both required fields
        if all(k in record for k in ['avg_bpm', 'body_temp']):
            processed_data.append(record)
    
    df = pd.DataFrame(processed_data)
    if len(df) > 0:
        print(f"Processed {len(df)} complete records with averaged BPM and body temperature")
        # Show data distribution
        print("\nData summary:")
        print(df.describe())
        
        # Show distribution of activities
        print("\nActivity distribution:")
        print(df['activity_id'].value_counts())
    else:
        print("No complete records found after processing")
    
    return df

def load_or_train_model():
    """Load the pretrained model or train a new one if not available."""
    try:
        # Try to load the saved model and preprocessing objects
        model = tf.keras.models.load_model('activity_model_bpm_temp.keras')
        encoder = joblib.load('activity_encoder_bpm_temp.joblib')
        
        # Create new scalers since we'll be using our own scaling approach
        bpm_scaler = StandardScaler()
        temp_scaler = StandardScaler()
        scaler = {'bpm': bpm_scaler, 'temp': temp_scaler}
        
        print("Loaded pre-trained model and created new scalers")
        return model, encoder, scaler
    except (OSError, IOError, ValueError) as e:
        print(f"Error loading pre-trained model: {e}")
        print("Training a new model...")
        return train_activity_model()

def process_mongodb_data_for_model(df, scaler):
    """Process the MongoDB data for model prediction."""
    # Check if we have all required features
    required_features = ['avg_bpm', 'body_temp']
    if not all(feature in df.columns for feature in required_features):
        missing = [feat for feat in required_features if feat not in df.columns]
        print(f"Missing required features in MongoDB data: {missing}")
        return None
    
    # Extract features
    X_bpm = df['avg_bpm'].values.reshape(-1, 1)
    X_temp = df['body_temp'].values.reshape(-1, 1)
    
    # Apply scaling to each feature separately
    X_bpm_scaled = scaler['bpm'].transform(X_bpm)
    X_temp_scaled = scaler['temp'].transform(X_temp)
    
    # Combine the scaled features
    X_scaled = np.hstack((X_bpm_scaled, X_temp_scaled))
    
    print(f"Feature shape after scaling: {X_scaled.shape}")
    return X_scaled

def predict_activities_from_mongodb(df, model, encoder):
    """Predict activities based on the MongoDB data."""
    if df is None or len(df) == 0:
        print("No data available for prediction")
        return None
    
    # Create a new scaler for each feature
    bpm_scaler = StandardScaler()
    temp_scaler = StandardScaler()
    
    # Fit scalers on the MongoDB data
    bpm_scaler.fit(df['avg_bpm'].values.reshape(-1, 1))
    temp_scaler.fit(df['body_temp'].values.reshape(-1, 1))
    
    # Create a combined scaler dictionary
    scaler = {'bpm': bpm_scaler, 'temp': temp_scaler}
    
    # Process data for model
    X_scaled = process_mongodb_data_for_model(df, scaler)
    if X_scaled is None:
        return None
    
    # Print some diagnostic information
    print(f"Model input shape: {model.input_shape}")
    print(f"Data shape for prediction: {X_scaled.shape}")
    
    # Make predictions
    predictions = model.predict(X_scaled)
    predicted_indices = np.argmax(predictions, axis=1)
    
    # Convert indices to activity labels
    activity_labels = encoder.categories_[0]
    decoded_predictions = [activity_labels[i] for i in predicted_indices]
    
    return decoded_predictions

def compare_predictions_with_actual(df, predicted_activities):
    """Compare the predicted activities with the actual activities from MongoDB."""
    if df is None or predicted_activities is None:
        return None
    
    # Create a comparison DataFrame
    comparison = pd.DataFrame({
        'Timestamp': df['timestamp'],
        'Actual Activity': df['activity_id'],
        'Predicted Activity': predicted_activities,
        'Match': df['activity_id'] == predicted_activities,
        'BPM': df['avg_bpm'],
        'Body Temp': df['body_temp']
    })
    
    # Calculate accuracy
    accuracy = comparison['Match'].mean() * 100
    print(f"\nPrediction Accuracy: {accuracy:.2f}%")
    
    # Create a confusion matrix
    actual_activities = df['activity_id'].values
    unique_activities = sorted(set(actual_activities) | set(predicted_activities))
    
    cm = confusion_matrix(
        actual_activities, 
        predicted_activities, 
        labels=unique_activities
    )
    
    print("\nConfusion Matrix:")
    cm_df = pd.DataFrame(cm, index=unique_activities, columns=unique_activities)
    print(cm_df)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(actual_activities, predicted_activities, labels=unique_activities))
    
    return comparison

def main():
    """Main function to run the entire workflow."""
    print("Starting the activity prediction workflow with averaged BPM and body temperature...")
    
    # Step 1: Load or train the model
    print("\n--- Step 1: Loading/Training the Model ---")
    model, encoder, scaler = load_or_train_model()
    if model is None:
        print("Failed to load or train model. Exiting.")
        return
    
    # Step 2: Fetch data from MongoDB
    print("\n--- Step 2: Fetching Data from MongoDB ---")
    mongo_data = fetch_sensor_data()
    if mongo_data is None:
        print("Failed to fetch data from MongoDB. Exiting.")
        return
    
    # Step 3: Process the MongoDB data - averaging pulse_BPM and BRPM
    print("\n--- Step 3: Processing MongoDB Data with Averaged BPM ---")
    mongo_df = process_sensor_data(mongo_data)
    if mongo_df is None or len(mongo_df) == 0:
        print("No valid data processed from MongoDB. Exiting.")
        return
    
    # Step 4: Predict activities using the trained model
    print("\n--- Step 4: Predicting Activities ---")
    predicted_activities = predict_activities_from_mongodb(mongo_df, model, encoder)
    if predicted_activities is None:
        print("Failed to predict activities. Exiting.")
        return
    
    # Step 5: Compare predictions with actual activities
    print("\n--- Step 5: Comparing Predictions with Actual Activities ---")
    comparison = compare_predictions_with_actual(mongo_df, predicted_activities)
    if comparison is not None:
        print("\nSample of Comparison Results:")
        print(comparison.head(10))
        
        # Save the comparison to a CSV file
        comparison.to_csv('activity_prediction_comparison_bpm_temp.csv', index=False)
        print("\nComparison results saved to 'activity_prediction_comparison_bpm_temp.csv'")
    
    print("\nWorkflow completed successfully!")

if __name__ == "__main__":
    main()
