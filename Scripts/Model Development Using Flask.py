from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load your trained machine learning model (replace 'model.pkl' with your model file)
model = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    prediction = model.predict(data)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
