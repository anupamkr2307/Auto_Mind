from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load model and data
try:
    model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
    car = pd.read_csv('Cleaned_Car.csv')
    print("Model and data loaded successfully!")
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    model = None
    car = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_page')
def predict_page():
    if car is None:
        return "Data not loaded properly", 500
    
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    years = sorted(car['year'].unique(), reverse=True)
    fuel_types = car['fuel_type'].unique()
    
    return render_template('predict.html', 
                         companies=companies, 
                         car_models=car_models, 
                         years=years, 
                         fuel_types=fuel_types)

@app.route('/get_models/<company>')
def get_models(company):
    if car is None:
        return jsonify([])
    
    models = car[car['company'] == company]['name'].unique().tolist()
    return jsonify(sorted(models))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get form data
        company = request.form.get('company')
        car_model = request.form.get('car_model')
        year = int(request.form.get('year'))
        fuel_type = request.form.get('fuel_type')
        kms_driven = int(request.form.get('kms_driven'))
        
        # Create prediction dataframe - match your original model's format
        prediction_data = pd.DataFrame({
            'name': [car_model],
            'company': [company],
            'year': [year],
            'kms_driven': [kms_driven],
            'fuel_type': [fuel_type]
        })
        
        # Make prediction
        prediction = model.predict(prediction_data)
        predicted_price = round(prediction[0], 2)
        
        return jsonify({
            'success': True,
            'predicted_price': predicted_price,
            'formatted_price': f"₹{predicted_price:,.2f}"
        })
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/services')
def services():
    return render_template('services.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(debug=True)