import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def process_df(df):
    """
    Basic processing:
    - fills numeric NA with median
    - label-encodes non-numeric columns
    Returns processed_df, encoders_dict
    """
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())

    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    encoders = {}
    for c in cat_cols:
        le = LabelEncoder()
        df[c] = df[c].astype(str)
        df[c] = le.fit_transform(df[c])
        encoders[c] = le

    return df, encoders

def split_vals(df, val_pct=0.2, target_col=None):
    """
    If target_col provided returns X_train, X_val, y_train, y_val
    Otherwise returns train/test split of df
    """
    if target_col and target_col in df.columns:
        y = df[target_col]
        X = df.drop(columns=[target_col])
        return train_test_split(X, y, test_size=val_pct, random_state=42)
    return train_test_split(df, test_size=val_pct, random_state=42)
