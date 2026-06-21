from app import get_data, predict, train
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Train")
    x_to_train = st.number_input(key="train_x", label="X", step=1, value=None)
    y_to_train = st.number_input(key="train_y", label="Y", value=None)
    data_to_train_filled = all(f is not None for f in [x_to_train, y_to_train])
    if st.button("Train", disabled=not data_to_train_filled):
        train(x_to_train, y_to_train)
        st.success(f"Trained with x={x_to_train}, y={y_to_train}")

    st.subheader("Predict")
    x_to_predict = st.number_input(key="predict_x", label="X", step=1, value=None)
    if st.button("Predict", disabled=x_to_predict is None):
        y_prediction = predict(x_to_predict)
        st.write(f"Predicted y: {y_prediction}")

with col_right:
    st.subheader("Data")
    data = get_data()
    if data:
        xs = [float(k) for k in data.keys() if k != "x"]
        ys = [float(v) for v in data.values() if v != "y"]
        df = pd.DataFrame({"x": xs, "y": ys}).sort_values("x").reset_index(drop=True)
        st.dataframe(df, width='stretch', hide_index=True)

        st.subheader("Chart")
        fig, ax = plt.subplots()
        ax.scatter(df["x"], df["y"], label="Data points")
        if len(df) >= 2:
            coeffs = np.polyfit(df["x"], df["y"], 1)
            line_x = np.linspace(df["x"].min(), df["x"].max(), 200)
            line_y = np.polyval(coeffs, line_x)
            ax.plot(line_x, line_y, color="red", label=f"y = {coeffs[0]:.2f}x + {coeffs[1]:.2f}")
            ax.legend()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        st.pyplot(fig)
