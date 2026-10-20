"""
Train Linear Regression Model on Cleaned_Car.csv and save to LinearRegressionModel.pkl
"""
import os
import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'Cleaned_Car.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'LinearRegressionModel.pkl')

def train_and_save():
    print(f"Loading data from {CSV_PATH}...")
    car = pd.read_csv(CSV_PATH)
    if 'Unnamed: 0' in car.columns:
        car = car.drop(columns=['Unnamed: 0'])
    
    X = car[['name', 'company', 'year', 'kms_driven', 'fuel_type']]
    y = car['Price']
    
    # Extract unique categories
    companies = sorted(car['company'].unique().tolist())
    fuel_types = sorted(car['fuel_type'].unique().tolist())
    names = sorted(car['name'].unique().tolist())
    
    ohe = OneHotEncoder(categories=[names, companies, fuel_types], handle_unknown='ignore', sparse_output=False)
    
    column_trans = make_column_transformer(
        (ohe, ['name', 'company', 'fuel_type']),
        remainder='passthrough'
    )
    
    lr = LinearRegression()
    pipe = make_pipeline(column_trans, lr)
    
    print("Fitting model...")
    pipe.fit(X, y)
    
    r2 = pipe.score(X, y)
    print(f"Model R^2 score: {r2:.4f}")
    
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(pipe, f)
    print(f"Model saved successfully to {MODEL_PATH}")
    
    # Verification test
    test_df = pd.DataFrame([['Maruti Suzuki Swift', 'Maruti', 2019, 100, 'Petrol']], columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])
    pred = pipe.predict(test_df)[0]
    print(f"Sample prediction for Swift 2019: INR {pred:,.0f}")

if __name__ == '__main__':
    train_and_save()
