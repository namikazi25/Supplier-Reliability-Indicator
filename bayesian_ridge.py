import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the saved model
with open('supplier_reliability_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Prepare your new data (X_new) in the same format as the training data
# Make sure to scale the new data using the same scaler used for training
scaler = StandardScaler()
X_new_scaled = scaler.fit_transform(X_new)

# Make predictions
predictions = loaded_model.predict(X_new_scaled)