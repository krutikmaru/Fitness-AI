from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from categories import categories
from diet_plans import diet_plans
import random
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

def determine_category(user_input):
    max_similarity = 0
    matched_category = None
    vectorizer = TfidfVectorizer()
    user_input_vector = vectorizer.fit_transform([user_input])
    for category, keywords in categories.items():
        category_keywords = ' '.join(keywords)
        category_vector = vectorizer.transform([category_keywords])
        similarity_score = cosine_similarity(user_input_vector, category_vector)[0][0]
        if similarity_score > max_similarity:
            max_similarity = similarity_score
            matched_category = category
    return matched_category

def get_random_diet_plan(category):
    return random.choice(diet_plans[category])

@app.route('/predict-diet-plan', methods=['POST'])
@cross_origin()
def predict_diet_plan():
    try:
        data = request.get_json(force=True)
        user_prompt = data.get('user_prompt', '')
        category = determine_category(user_prompt)
        if category:
            diet_plan = get_random_diet_plan(category)
            response = {
                'status': 'success',
                'category': category,
                'diet_plan': {
                    'breakfast': diet_plan.split('\n')[0],
                    'lunch': diet_plan.split('\n')[1],
                    'dinner': diet_plan.split('\n')[2]
                }
            }
        else:
            response = {'status': 'failure', 'message': "Sorry, I couldn't understand your request"}
        return jsonify(response)
    except Exception as e:
        return jsonify({'status': 'failure', 'error': str(e)})

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
