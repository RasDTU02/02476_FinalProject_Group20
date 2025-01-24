# data_drift.py
import pandas as pd
import numpy as np
from pathlib import Path
# from sklearn.model_selection import train_test_split
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.pipeline.column_mapping import ColumnMapping

def create_simulated_drift_data(reference_data, current_data, target_column):
    # Simulate Feature Drift
    # we can simulate a form of drift by randomly changing the image paths.
    # For example, we can add a random string to the image path to simulate a change in data distribution.
    current_data['image_path'] = current_data['image_path'].apply(lambda x: x + f'_{np.random.randint(1, 1000)}')
    
    # Simulate Label Drift
    # We change some labels to simulate a change in class distribution.
    unique_labels = current_data[target_column].unique()
    mask = np.random.choice([True, False], size=len(current_data), p=[0.1, 0.9])  # 10 % chance of editing the label
    current_data.loc[mask, target_column] = np.random.choice(unique_labels, size=sum(mask))
    
    return reference_data, current_data

def perform_data_drift(reference_data, current_data, target_column):
    # Setup column mapping
    column_mapping = ColumnMapping()
    column_mapping.target = target_column

    # Create and run report
    report = Report(metrics=[DataDriftPreset()])
    
    report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)
    
    # Save report
    report_path = Path('data_drift')
    report_path.mkdir(parents=True, exist_ok=True)
    report.save_html(str(report_path / 'data_drift_report.html'))

if __name__ == "__main__":
    # Load data
    reference_data = pd.read_csv('/Users/bsm/Desktop/02476_FinalProject_Group20/data/csv_files/train.csv')
    current_data = pd.read_csv('/Users/bsm/Desktop/02476_FinalProject_Group20/data/csv_files/test.csv')
    target_column = 'label'
    
    # Create simulated drift data
    reference_data, current_data = create_simulated_drift_data(reference_data, current_data, target_column)
    
    # Perform drift analysis
    perform_data_drift(reference_data, current_data, target_column)