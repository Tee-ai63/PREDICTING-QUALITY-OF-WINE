# train_and_save_model.py
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Load and prepare data
def prepare_data():
    wine_data = pd.read_csv('WineQT.csv')
    
    # Remove Id column if exists
    if 'Id' in wine_data.columns:
        wine_data = wine_data.drop('Id', axis=1)
    
    return wine_data

def engineer_features(X_df):
    """Add domain-knowledge features for wine quality prediction"""
    # Create new features based on wine chemistry knowledge
    X_df['acid_balance'] = X_df['fixed acidity'] / (X_df['pH'] + 1e-6)
    X_df['alcohol_acidity_ratio'] = X_df['alcohol'] / (X_df['fixed acidity'] + 1e-6)
    X_df['free_so2_ratio'] = X_df['free sulfur dioxide'] / (X_df['total sulfur dioxide'] + 1e-6)
    
    return X_df

def train_model():
    # Load data
    wine_data = prepare_data()
    
    # Prepare features and target
    X = wine_data.drop('quality', axis=1)
    y = wine_data['quality']
    
    # Engineer features
    X_eng = engineer_features(X.copy())
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_eng)
    
    # Train RandomForest model (or your best model from analysis)
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_scaled, y)
    
    # Save the model and scaler
    with open('wine_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save feature names
    with open('feature_names.pkl', 'wb') as f:
        pickle.dump(X_eng.columns.tolist(), f)
    
    print("Model training complete!")
    print(f"Model saved: wine_model.pkl")
    print(f"Scaler saved: scaler.pkl")
    print(f"Feature names saved: feature_names.pkl")

if __name__ == "__main__":
    train_model()