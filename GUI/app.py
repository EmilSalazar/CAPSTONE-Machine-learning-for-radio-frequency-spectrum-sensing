from flask import Flask, render_template, jsonify, session
import pandas as pd
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__, static_folder='static')
app.secret_key = 'your_secret_key'  # Needed to use sessions

# Change the path to the data to where its located on your computer
model = load_model('C:\\Users\\Emil\\PycharmProjects\\CapstoneKerasML\\Testing\\trained_model.h5') # Load saved model

# Load new dataset
new_data_path = 'C:\\Users\\Emil\\PycharmProjects\\CapstoneKerasML\\Testing\\RandomizedData.csv'
new_dataset = pd.read_csv(new_data_path)

# Separate features
X_new = new_dataset.iloc[:, :-1].values
y_new = new_dataset.iloc[:, -1].values  # All rows, last column (labels)

# Initialize index
index = 0

@app.route('/')
def index():
    session['index'] = 0  # Reset the index each time the home page is loaded
    return render_template('index.html')

@app.route('/predict')
def predict():
    # Retrieve current index
    idx = session.get('index', 0)

    # Make a prediction
    prediction = model.predict(np.expand_dims(X_new[idx], axis=0))
    predicted_class = np.argmax(prediction, axis=1)
    labels = {0: "Bluetooth", 1: "WiFi", 2: "Neither"}
    result = labels[predicted_class[0]]

    # Update index for next call
    idx = (idx + 1) % len(X_new)  # Wrap around if the index exceeds dataset length
    session['index'] = idx

    return jsonify(prediction=result)

if __name__ == '__main__':
    app.run(debug=True)