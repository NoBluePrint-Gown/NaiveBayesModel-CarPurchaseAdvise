"""
model.py - Car Evaluation Model Training
Trains a Naive Bayes classifier on the Car Evaluation dataset and saves the model and encoders.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# Define column names and their possible values
COLUMN_NAMES = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
TARGET_NAME = 'class'

# Define the order of categories for each feature (for ordinal encoding)
CATEGORIES = {
    'buying': ['vhigh', 'high', 'med', 'low'],
    'maint': ['vhigh', 'high', 'med', 'low'],
    'doors': ['2', '3', '4', '5more'],
    'persons': ['2', '4', 'more'],
    'lug_boot': ['small', 'med', 'big'],
    'safety': ['low', 'med', 'high']
}

def load_and_prepare_data(filepath='car.data'):
    """
    Load the car evaluation dataset and prepare it for training.
    
    Args:
        filepath (str): Path to the car.data file
        
    Returns:
        tuple: X (features), y (target), feature_encoder, target_encoder
    """
    print(f"Loading data from {filepath}...")
    
    # Load the data (car.data is space-separated or comma-separated)
    # First try to detect the separator
    with open(filepath, 'r') as f:
        first_line = f.readline().strip()
        separator = ',' if ',' in first_line else ' '
    
    df = pd.read_csv(filepath, sep=separator, header=None, names=COLUMN_NAMES + [TARGET_NAME])
    
    print(f"Dataset shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Prepare features and target
    X = df[COLUMN_NAMES]
    y = df[TARGET_NAME]
    
    # Create and fit encoders
    feature_encoder = OrdinalEncoder(categories=[CATEGORIES[col] for col in COLUMN_NAMES])
    target_encoder = LabelEncoder()
    
    # Encode features and target
    X_encoded = feature_encoder.fit_transform(X)
    y_encoded = target_encoder.fit_transform(y)
    
    print(f"\nTarget classes: {target_encoder.classes_}")
    print(f"Class distribution:")
    for class_name, count in zip(target_encoder.classes_, np.bincount(y_encoded)):
        print(f"  {class_name}: {count} ({count/len(y_encoded)*100:.1f}%)")
    
    return X_encoded, y_encoded, feature_encoder, target_encoder

def train_model(X_train, y_train):
    """
    Train a Categorical Naive Bayes model.
    
    Args:
        X_train: Training features
        y_train: Training target
        
    Returns:
        Trained model
    """
    print("\n" + "="*50)
    print("Training Categorical Naive Bayes Model...")
    print("="*50)
    
    # Create and train the model
    model = CategoricalNB()
    model.fit(X_train, y_train)
    
    print("Model training completed!")
    
    # Print class priors
    print("\nClass priors (log probabilities):")
    for i, prior in enumerate(model.class_log_prior_):
        print(f"  Class {i}: {np.exp(prior):.4f}")
    
    return model

def evaluate_model(model, X_test, y_test, target_encoder):
    """
    Evaluate the trained model on test data.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        target_encoder: Target label encoder
    """
    print("\n" + "="*50)
    print("Model Evaluation")
    print("="*50)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))

def save_model(model, feature_encoder, target_encoder, filepath='car_model.joblib'):
    """
    Save the trained model and encoders.
    
    Args:
        model: Trained Naive Bayes model
        feature_encoder: Fitted ordinal encoder for features
        target_encoder: Fitted label encoder for target
        filepath: Path to save the model
    """
    print("\n" + "="*50)
    print("Saving Model...")
    print("="*50)
    
    model_data = {
        'model': model,
        'feature_encoder': feature_encoder,
        'target_encoder': target_encoder,
        'column_names': COLUMN_NAMES,
        'categories': CATEGORIES
    }
    
    joblib.dump(model_data, filepath)
    print(f"Model saved successfully to '{filepath}'")
    print(f"File size: {os.path.getsize(filepath):,} bytes")

def main():
    """Main training function"""
    print("="*50)
    print("CAR EVALUATION MODEL TRAINING")
    print("="*50)
    
    # Check if car.data exists
    if not os.path.exists('car.data'):
        print("\nERROR: 'car.data' file not found!")
        print("Please download the Car Evaluation dataset from:")
        print("https://archive.ics.uci.edu/ml/datasets/Car+Evaluation")
        print("\nOr use this direct link:")
        print("https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data")
        return
    
    # Load and prepare data
    X, y, feature_encoder, target_encoder = load_and_prepare_data('car.data')
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Evaluate the model
    evaluate_model(model, X_test, y_test, target_encoder)
    
    # Save the model and encoders
    save_model(model, feature_encoder, target_encoder)
    
    print("\n" + "="*50)
    print("Training completed successfully!")
    print("="*50)

if __name__ == "__main__":
    main()