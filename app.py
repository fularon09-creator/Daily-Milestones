from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

with open('heart_disease_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('heart_disease_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return "Heart Disease Prediction API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['features']
    features = np.array(data).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]
    result = {
        'prediction': int(prediction),
        'diagnosis': 'Disease' if prediction == 1 else 'No Disease',
        'probability': round(float(probability), 3)
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)