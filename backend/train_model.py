import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import os

def train_and_save_model():
    print("Loading dataset...")
    # Assuming CSV is in the parent directory as uploaded by user
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'Crop_recommendation (1).csv')
    
    if not os.path.exists(csv_path):
        print(f"Dataset not found at {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    print("Dataset loaded successfully.")

    # Features and labels
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = df['label']

    # Label encoding
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Train model
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Convert test accuracy
    accuracy = model.score(X_test, y_test)
    print(f"Model trained. Validation Accuracy: {accuracy * 100:.2f}%")

    # Save model and encoder
    model_path = os.path.join(os.path.dirname(__file__), 'rf_model.pkl')
    le_path = os.path.join(os.path.dirname(__file__), 'label_encoder.pkl')
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
        
    with open(le_path, 'wb') as f:
        pickle.dump(label_encoder, f)
        
    print("Model and Label Encoder saved successfully to disk.")

if __name__ == "__main__":
    train_and_save_model()
