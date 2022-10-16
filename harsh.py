import streamlit as st
import numpy as np
import pandas as pd
import pickle

pipe = pickle.load(open('harsh.pkl', 'rb'))

st.header('Car Price Predictor')

year = st.number_input('Year of Manufacture ')

kms = st.number_input('KMs driven')

fuel = st.selectbox('Fuel Type', ('Diesel', 'Petrol'))

transmission = st.selectbox('Transmission', ('Manual', 'Automatic'))

owner = st.selectbox('Owner', ('First Owner', 'Second Owner', 'Third Owner'))

mileage = st.number_input('Mileage')

seats = st.number_input('Seats')

brand = st.selectbox('Brand', (
    'Maruti', 'Hyundai', 'Mahindra', 'Tata', 'Ford', 'Honda', 'Toyota', 'Renault', 'Chevrolet', 'Volkswagen'))

if st.button('Predict Price'):
    take_input = np.array([[year, kms, fuel, transmission, owner, mileage, seats, brand]])
    take_input = pd.DataFrame(take_input,
                              columns=['year', 'km_driven', 'fuel', 'transmission', 'owner', 'mileage', 'seats', 'brand'])

    y_pred = pipe.predict(take_input)
    st.title("Rs : " + str(np.round(y_pred[0])))
