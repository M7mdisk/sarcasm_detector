from flask import Flask, request, jsonify
from transformers import pipeline


classifier = pipeline("sentiment-analysis", model="sarca_detect/")

def get_prediction(text):
    pred = classifier(text)[0]
    label = "Sarcasm" if pred["label"] == 1 else "Real"
    return label,pred["score"]

app = Flask(__name__)

@app.route('/echo', methods=['POST']) 
def foo():
    data = request.json
    return jsonify(data)

@app.route('/predict', methods=['POST']) 
def predict():
    data = request.json
    string = data["text"]
    label,accuracy = get_prediction(string)

    return jsonify({"label":label,"probability":accuracy})

if __name__ == '__main__':
    app.run(debug=True)