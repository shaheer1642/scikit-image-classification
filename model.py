# built-in libraries
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# user-defined libraries
from data_processing import X_train, y_train

# Load the saved model if it exists, otherwise train a new model
model_filename = "rf_model.pkl"
if os.path.exists(model_filename):
    model = joblib.load(model_filename)
else:
    # Initialize a new Random Forest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    # Train the classifier
    model.fit(X_train, y_train)
    # Save the trained model
    joblib.dump(model, model_filename)
