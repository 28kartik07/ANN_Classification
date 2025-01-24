## creating the interface for user to enter the inputs and see the prediction
import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler,LabelEncoder, OneHotEncoder

## load the ann train model
model = tf.keras.models.load_model('model.h5')

## load the encoders and scaler
with open('label_encode_gender.pkl','rb') as file:
    label_encode_gender = pickle.load(file)

with open('onehotencoder_geo.pkl','rb') as file:
    onehotencoder_geo = pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)

## creating a streamlit application 
st.title('Customer Churn Prediction')

## user input 
geography = st.selectbox('Geography', onehotencoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encode_gender.classes_)
age = st.slider('Age',18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
esitmated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure',0,10)
num_of_products = st.slider('Number of Products',1,4)
has_cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_member = st.selectbox('Is Active Member',[0,1])

## convert data to a object
input_data = {
    'CreditScore' : credit_score,
    'Geography' : geography,
    'Gender' : label_encode_gender.transform([gender]),
    'Age' : age,
    'Tenure' : tenure,
    'Balance' : balance,
    'NumOfProducts' : num_of_products,
    'HasCrCard' : has_cr_card,
    'IsActiveMember' : is_active_member,
    'EstimatedSalary' : esitmated_salary
}

input_data = pd.DataFrame([input_data])
## encoding the geography values to onehot encoding
geo_encoder = onehotencoder_geo.transform(input_data[['Geography']]).toarray()
geo_encoded_columns = pd.DataFrame(geo_encoder,columns=onehotencoder_geo.get_feature_names_out(['Geography']))

## concatenating the ohe data to input data
input_data = pd.concat([input_data.drop(['Geography'],axis=1),geo_encoded_columns],axis=1)

## scaling the input data
input_scaled = scaler.transform(input_data)

## making the prediction 
prediction = model.predict(input_scaled)
prediction_prob = prediction[0][0]

## stating the result
st.write(f"The probability is : {prediction_prob}")
if prediction_prob > 0.5:
    st.write("The user will Churn")
else:
    st.write("The user will not Churn")
   