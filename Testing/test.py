import pandas as pd
from tensorflow.keras.models import load_model
import numpy as np

# Change the path to the data to where its located on your computer
model = load_model('C:\\Users\\Emil\\PycharmProjects\\CapstoneKerasML\\Testing\\trained_model.h5') # Load saved model

# Load new dataset
new_data_path = 'C:\\Users\\Emil\\PycharmProjects\\CapstoneKerasML\\Testing\\2024newdata.csv'
new_dataset = pd.read_csv(new_data_path)

# Separate features and labels
X_new = new_dataset.iloc[:, :-1].values  # All rows, all columns except the last one (features)
y_new = new_dataset.iloc[:, -1].values  # All rows, last column (labels)

# Make guess with the model
predictions = model.predict(X_new)
predicted_classes = np.argmax(predictions, axis=1)

# labels for printing
labels = {0: "Bluetooth", 1: "WiFi", 2: "Neither"}

# Print out guess and actual labels
for i, (prediction, actual) in enumerate(zip(predicted_classes, y_new)):
    print(f"Row {i+1}: Model's guess - {labels[prediction]}, Actual - {labels[actual]}")
