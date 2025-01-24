# data_drift.py
import random
import time
from pathlib import Path

import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import (DataDriftPreset, DataQualityPreset,
                                     TargetDriftPreset)
from evidently.report import Report
from tqdm import tqdm

# Assume the script runs from src/rice/
csv_files_path = Path('data/csv_files')
raw_data_path = Path('data/raw')

def create_reference_data():
    """Create reference_data.csv from train.csv"""
    print("Creating reference data...")
    train_data = pd.read_csv(csv_files_path / 'train.csv')
    
    # Ensure 'target' column exists in reference data
    if 'target' not in train_data.columns:
        # Check for common alternative names for target column
        possible_target_names = ['label', 'class', 'category', 'y']
        for name in possible_target_names:
            if name in train_data.columns:
                train_data = train_data.rename(columns={name: 'target'})
                print(f"Renamed '{name}' to 'target' in reference data.")
                break
        else:
            # If no common target column name found, we need to decide or raise an error
            print("No common target column found in reference data. Assuming 'label' is the target.")
            if 'label' in train_data.columns:
                train_data = train_data.rename(columns={'label': 'target'})
            else:
                raise ValueError("The 'target' column or any common alternative is not found in the reference data. Please check your data structure.")
    
    train_data.to_csv(csv_files_path / 'reference_data.csv', index=False)
    print("Reference data has been created as 'reference_data.csv' in csv_files folder.")

def create_current_data():
    """Create current_data.csv from prediction_database.csv if it exists, otherwise simulate data"""
    prediction_database_path = raw_data_path / 'prediction_database.csv'
    if prediction_database_path.exists():
        print("Creating current data from existing prediction data...")
        current_data = pd.read_csv(prediction_database_path)
        # Ensure 'target' column exists in current data
        if 'prediction' in current_data.columns:
            current_data = current_data.rename(columns={'prediction': 'target'})
        elif 'target' not in current_data.columns:
            raise ValueError("The 'target' column or 'prediction' column is not found in the current data.")
        current_data.to_csv(csv_files_path / 'current_data.csv', index=False)
        print("Current data has been created as 'current_data.csv' in csv_files folder from existing prediction data.")
    else:
        print("Simulating prediction data since 'prediction_database.csv' does not exist.")
        num_predictions = 10  # Adjust this number based on how many entries you want
        
        predictions = []
        for i in tqdm(range(num_predictions), desc="Simulating Predictions"):
            # Simulating time
            time.sleep(0.1)  # Simulate some delay to see the progress bar in action
            current_time = pd.Timestamp.now()  # Changed from 'time' to 'current_time'
            
            # Simulating image paths
            image_path = f'data/raw/Rice_Image_Dataset/Class_{i % 5}/rice_image_{i}.jpg'
            
            # Simulating prediction
            prediction = random.randint(0, 4)  # Assuming 5 classes for rice types
            
            predictions.append({
                'time': current_time,  # Changed from 'time' to 'current_time'
                'image_path': image_path,
                'prediction': prediction
            })
        
        # Convert predictions to DataFrame and save
        df = pd.DataFrame(predictions)
        df = df.rename(columns={'prediction': 'target'})  # Rename 'prediction' to 'target'
        df.to_csv(prediction_database_path, index=False)
        df.to_csv(csv_files_path / 'current_data.csv', index=False)
        print(f"Simulated prediction data saved to {prediction_database_path} and created 'current_data.csv'.")

# Create reference_data.csv if it does not exist
if not (csv_files_path / 'reference_data.csv').exists():
    create_reference_data()

# Create current_data.csv if it does not exist
if not (csv_files_path / 'current_data.csv').exists():
    create_current_data()

# Load reference and current data
print("Loading reference data...")
reference_data = pd.read_csv(csv_files_path / 'reference_data.csv')
print("Loading current data...")
current_data = pd.read_csv(csv_files_path / 'current_data.csv')

# Define column mapping
print("Setting up column mapping...")
column_mapping = ColumnMapping()
column_mapping.target = 'target'  # Specify which column is the target
column_mapping.prediction = 'target'  # We set prediction to target to avoid confusion

# Perform drift analysis
print("Performing drift analysis...")
report = Report(metrics=[
    DataDriftPreset(),
    DataQualityPreset(),
    TargetDriftPreset()
])

# Run the report with column mapping
try:
    start_time = time.time()
    report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)
    end_time = time.time()
    print(f"Drift analysis completed in {end_time - start_time:.2f} seconds")
except ValueError as e:
    print(f"An error occurred: {e}")
    print("Please ensure both datasets have the 'target' column or adjust the script according to your data structure.")
    raise

# Save the report as an HTML file
print("Saving the drift analysis report...")
report_path = Path('reports')
report_path.mkdir(exist_ok=True)
report.save_html(report_path / 'data_drift_report.html')

print(f"Drift analysis report has been saved to {report_path / 'data_drift_report.html'}")