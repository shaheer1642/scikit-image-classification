from flask import Flask, request, jsonify
from image_detection import imageDetectionBase64

app = Flask(__name__)

@app.route('/predict', methods=['POST']) 
def predict():
    data = request.json
    res = imageDetectionBase64(data['image'])
    return jsonify(
        message= res[0],
        code= 200
    )