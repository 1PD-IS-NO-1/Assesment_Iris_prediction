from flask import Flask, render_template, request, jsonify
from model import load_model, predict_class


app = Flask(__name__)
model, label_map = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        sepal_length = float(data['sepal_length'])
        sepal_width = float(data['sepal_width'])

        prediction = predict_class(model, label_map, sepal_length, sepal_width)
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run()
