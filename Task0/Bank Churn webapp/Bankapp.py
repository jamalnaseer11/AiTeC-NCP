# app.py
from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('customer_churn_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extracting information from the form
    features = [x for x in request.form.values()]
    
    # Convert features to appropriate types
    credit_score = float(features[0])
    city = features[1]
    gender = 1 if features[2].lower() == 'male' else 0
    age = float(features[3])
    tenure = float(features[4])
    balance = float(features[5])
    num_of_products = float(features[6])
    has_credit_card = float(features[7])
    is_active_member = float(features[8])
    estimated_salary = float(features[9])
    
    # One-hot encoding for city
    cities = ['Spain', 'Germany']  # Assuming 'City1' is the dropped column
    city_encoding = [0, 0]
    if city in cities:
        city_encoding[cities.index(city)] = 1
    
    # Combine all features
    input_features = [credit_score] + city_encoding + [gender, age, tenure, balance, num_of_products, has_credit_card, is_active_member, estimated_salary]
    input_features = np.array(input_features).reshape(1, -1)
    
    print(input_features)
    
    
    #scaling
    input_features=scaler.transform(input_features)
    
    
    
    print(input_features)
   
    
    
  
    # Make prediction
    prediction = model.predict(input_features)
    print ('prediction=',prediction)
    
    # Render results
    if prediction >0.5:
        result = "The customer is likely to churn."
    else:
        result = "The customer is not likely to churn."
        
    return render_template('result.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)
