# app.py

# import streamlit as st
# import numpy as np
# import joblib

# # Load saved objects
# model = joblib.load("model.pkl")
# scaler = joblib.load("scaler.pkl")

# st.title("🎓 Student Marks Predictor")

# st.write("Enter student details below:")

# attendance = st.slider("Attendance (%)", 0.0, 100.0, 75.0)
# previous_grades = st.slider("Previous Grades", 0.0, 100.0, 60.0)

# if st.button("Predict Final Marks"):
#     try:
#         input_data = np.array([[study_hours, attendance, previous_grades]])
#         input_scaled = scaler.transform(input_data)
#         prediction = model.predict(input_scaled)

#         st.success(f"Predicted Final Marks: {prediction[0]:.2f}")

#     except Exception as e:
#         st.error(f"Error: {e}")
import streamlit as st
import numpy as np
import joblib

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🎓 Student Marks Predictor")

# ✅ Same naam everywhere use karo
study_hours = st.slider("Study Hours per Day", 0.0, 12.0, 5.0)
attendance = st.slider("Attendance (%)", 0.0, 100.0, 75.0)
previous_grades = st.slider("Previous Grades", 0.0, 100.0, 60.0)

if st.button("Predict Final Marks"):

    input_data = np.array([[study_hours, attendance, previous_grades]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    st.success(f"Predicted Final Marks: {prediction[0]:.2f}")


