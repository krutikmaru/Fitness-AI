from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import joblib

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

calories_model = joblib.load('linear_regression_model.joblib')
water_model = joblib.load('logistic_regression_water_model.joblib')
steps_distance_model = joblib.load('linear_regression_distance_model.joblib')
steps_walking_time_model = joblib.load('linear_regression_walking_time_model.joblib')

@app.route('/')
@cross_origin()
def hello_world():
    return 'Hello from Flask!'


@app.route('/predict-weight-change', methods=['POST'])
@cross_origin()
def predict_weight_change():
    try:
        data = request.get_json(force=True)
        calories_burnt = float(data.get('value', 0))
        prediction = calories_model.predict([[calories_burnt]])[0]
        return jsonify({'prediction': round(prediction, 2)})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/predict-hydration', methods=['POST'])
@cross_origin()
def predict_hydration():
    try:
        data = request.get_json(force=True)
        water_consumed = float(data.get('value', 0))
        prediction = water_model.predict([[water_consumed]])[0]
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})
    
@app.route('/predict-distance', methods=['POST'])
@cross_origin()
def predict_distance():
    try:
        data = request.get_json(force=True)
        steps = float(data.get('value', 0))
        prediction = steps_distance_model.predict([[steps]])[0]
        return jsonify({'prediction': round(prediction, 2)})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/predict-walking-time', methods=['POST'])
@cross_origin()
def predict_walking_time():
    try:
        data = request.get_json(force=True)
        steps = float(data.get('value', 0))
        prediction = steps_walking_time_model.predict([[steps]])[0]
        return jsonify({'prediction': round(prediction, 2)})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
