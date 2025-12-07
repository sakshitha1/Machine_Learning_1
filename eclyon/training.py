# eclyon/training.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_model(X_train, y_train):
    """
    Train a Random Forest classifier
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_val, y_val):
    """
    Evaluate the model on validation data
    """
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    return accuracy

def predict(model, X):
    """
    Predict using trained model
    """
    return model.predict(X)
