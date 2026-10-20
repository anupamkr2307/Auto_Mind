import sys
from api.index import app

client = app.test_client()

routes = [
    ('/', 'Auto_Mind - complete car solution'),
    ('/predict_page', 'Car Price Prediction - Auto_Mind - complete car solution'),
    ('/about', 'About - Auto_Mind - complete car solution'),
    ('/services', 'Services - Auto_Mind - complete car solution'),
    ('/contact', 'Contact - Auto_Mind - complete car solution'),
    ('/chatbot', 'ChatBot - Auto_Mind - complete car solution'),
    ('/connect', 'Connect - Auto_Mind - complete car solution'),
    ('/connect/signup', 'Sign Up - Connect | Auto_Mind - complete car solution'),
    ('/connect/login', 'Login - Connect | Auto_Mind - complete car solution'),
]

all_passed = True
for path, expected_title in routes:
    res = client.get(path)
    if res.status_code != 200:
        print(f"FAIL {path}: status code {res.status_code}")
        all_passed = False
    elif expected_title not in res.get_data(as_text=True):
        print(f"FAIL {path}: expected title '{expected_title}' not found in HTML")
        all_passed = False
    else:
        print(f"PASS {path}: 200 OK and title matched")

# Test car price predictions
test_cases = [
    ('Hyundai', 'Hyundai Santro Xing', 2014, 25000, 'Petrol'),
    ('Maruti', 'Maruti Suzuki Swift', 2018, 40000, 'Petrol'),
    ('Ford', 'Ford Figo', 2013, 60000, 'Diesel')
]

for company, car_model, year, kms, fuel in test_cases:
    res = client.post('/predict', data={
        'company': company,
        'car_model': car_model,
        'year': year,
        'kms_driven': kms,
        'fuel_type': fuel
    })
    json_data = res.get_json()
    if json_data.get('success') and json_data.get('predicted_price'):
        price = json_data['predicted_price']
        print(f"PASS Predict: {car_model} ({year}) -> INR {price:.2f}")
    else:
        print(f"FAIL Predict for {car_model}: {json_data}")
        all_passed = False

# Test chat
chat_res = client.post('/chat', json={'message': 'How do I know my car resale value?'})
c_json = chat_res.get_json()
if c_json.get('success') and len(c_json.get('response', '')) > 20:
    print("PASS Chat response generated successfully")
else:
    print(f"FAIL Chat response: {c_json}")
    all_passed = False

if all_passed:
    print("\n>>> ALL TESTS PASSED SUCCESSFULLY! <<<")
else:
    print("\n>>> SOME TESTS FAILED! <<<")
    sys.exit(1)
