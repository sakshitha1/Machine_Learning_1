# eclyon/preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """
    Load CSV file into a pandas DataFrame
    """
    df = pd.read_csv(file_path)
    return df

def clean_data(df):
    """
    Simple preprocessing: drop missing values and duplicates
    """
    df = df.dropna()
    df = df.drop_duplicates()
    return df

def feature_target_split(df, target_column):
    """
    Split DataFrame into features X and target y
    """
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    return X, y

def train_val_split(X, y, test_size=0.2, random_state=42):
    """
    Split data into train and validation sets
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def scale_features(X_train, X_val):
    """
    Standardize features (zero mean, unit variance)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    return X_train_scaled, X_val_scaled, scaler
