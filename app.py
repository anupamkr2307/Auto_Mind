import os
import pickle
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
from markdown import markdown
import re

# Import configuration
try:
    from config import AK, APP_HOST, APP_PORT
except ImportError:
    # Fallback to environment variables if config.py doesn't exist
    AK = os.getenv('AK')
    APP_HOST = '0.0.0.0'
    APP_PORT = 5000

app = Flask(__name__)

# Load the trained model
try:
    model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Load the cleaned dataset to get unique values for dropdowns
try:
    df = pd.read_csv('Cleaned_Car.csv')
    print("Dataset loaded successfully!")
    
    # Get unique values for dropdowns
    companies = sorted(df['company'].unique().tolist())
    fuel_types = sorted(df['fuel_type'].unique().tolist())
    years = sorted(df['year'].unique().tolist())
    
    # Get models for each company
    company_models = {}
    for company in companies:
        company_models[company] = sorted(df[df['company'] == company]['name'].unique().tolist())
    
except Exception as e:
    print(f"Error loading dataset: {e}")
    companies = []
    fuel_types = []
    years = []
    company_models = {}

# Configure AI API
if AK and AK != "your_ai_api_key_here":
    genai.configure(api_key=AK)
    ai_model = genai.GenerativeModel('gemini-pro-latest')
    print("AI API configured successfully!")
else:
    print("AK not configured. Chatbot functionality will be limited.")
    print("To enable chatbot: Edit config.py and add your AI API key")
    ai_model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_page')
def predict_page():
    return render_template('predict.html', 
                         companies=companies, 
                         fuel_types=fuel_types, 
                         years=years)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/services')
def services():
    return render_template('services.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/get_models/<company>')
def get_models(company):
    """Get models for a specific company"""
    if company in company_models:
        return jsonify(company_models[company])
    return jsonify([])

@app.route('/predict', methods=['POST'])
def predict():
    """Handle car price prediction"""
    try:
        if model is None:
            return jsonify({'success': False, 'error': 'Model not loaded'})
        
        # Get form data
        company = request.form.get('company')
        car_model = request.form.get('car_model')
        year = int(request.form.get('year'))
        kms_driven = int(request.form.get('kms_driven'))
        fuel_type = request.form.get('fuel_type')
        
        # Validate inputs
        if not all([company, car_model, year, kms_driven, fuel_type]):
            return jsonify({'success': False, 'error': 'All fields are required'})
        
        # Create prediction input
        prediction_data = pd.DataFrame([[
            car_model,
            company,
            year,
            kms_driven,
            fuel_type
        ]], columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])
        
        # Make prediction
        predicted_price = model.predict(prediction_data)[0]
        
        # Format the price
        formatted_price = f"₹{predicted_price:,.0f}"
        
        return jsonify({
            'success': True,
            'predicted_price': predicted_price,
            'formatted_price': formatted_price
        })
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chatbot messages"""
    try:
        if ai_model is None:
            return jsonify({
                'success': False, 
                'response': 'Chatbot service is currently unavailable. Please set the AK environment variable.'
            })
        
        data = request.get_json()
        message = data.get('message', '')
        
        if not message:
            return jsonify({'success': False, 'response': 'Please provide a message.'})
        
        # Create a context-aware prompt for car-related queries
        prompt = f"""
        You are an AI assistant for Auto_Mind, a car price prediction platform. 
        You help users with car-related questions, pricing information, vehicle specifications, 
        buying and selling advice, car maintenance tips, and general automotive knowledge.
        
        User's question: {message}
        
        Please provide a helpful, accurate, and detailed response. If the question is about car pricing,
        you can mention that Auto_Mind provides AI-powered price predictions. Keep your response 
        conversational and informative.
        """
        
        # Generate response using AI
        response = ai_model.generate_content(prompt)
        
        # Convert markdown to HTML
        html_response = markdown(response.text)
        
        return jsonify({
            'success': True,
            'response': html_response
        })
        
    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({
            'success': False, 
            'response': 'Sorry, I encountered an error. Please try again later.'
        })

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', APP_PORT))
    app.run(debug=False, host='0.0.0.0', port=port)
