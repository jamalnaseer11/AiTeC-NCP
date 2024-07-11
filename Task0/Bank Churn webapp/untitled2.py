import joblib
import numpy as np

model = joblib.load('customer_churn_model.pkl')




credit_score = 619
city = 'France'
gender = 1
age = 42
tenure =2
balance = 31431
num_of_products = 2
has_credit_card = 1
is_active_member = 1
estimated_salary = 1124124


   # One-hot encoding for city
cities = ['Spain', 'Germany']  # Assuming 'City1' is the dropped column
city_encoding = [0, 0]
if city in cities:
 city_encoding[cities.index(city)] = 1




# Combine all features
input_features = [credit_score] + city_encoding + [gender, age, tenure, balance, num_of_products, has_credit_card, is_active_member, estimated_salary]
input_features = np.array(input_features).reshape(1, -1)
   
print(input_features)
   
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
input_features = sc_X.fit_transform(input_features)
   
   
   
print(input_features)
  