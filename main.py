from app import predict
from app import train
import streamlit as st

x_to_train = st.number_input("X", value=None)
y_to_train = st.number_input("Y", value=None)

data_to_train_filled = all(f is not None for f in [x_to_train, y_to_train])
if st.button("Train", disabled=not data_to_train_filled):
    train(x_to_train, y_to_train)

x_to_predict = st.number_input("X to predict", step=1, value=None)
if st.button("Predict", disabled=x_to_predict is None):
    y_prediction = predict(x_to_predict)
    st.write(y_prediction)