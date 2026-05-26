import streamlit as st
import pandas as pd
import joblib
import time

# ─────────────────────────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────────────────────────

model = joblib.load("models/student_model.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

# ─────────────────────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Student Performance Prediction")

st.write("Enter student details below.")

# ─────────────────────────────────────────────────────────────
# Inputs
# ─────────────────────────────────────────────────────────────

study_hours = st.slider("Study Hours", 0, 12, 5)

attendance_pct = st.slider("Attendance %", 0, 100, 80)

prev_gpa = st.slider("Previous GPA", 0.0, 4.0, 3.0)

assignments_done = st.slider("Assignments Done %", 0, 100, 75)

sleep_hours = st.slider("Sleep Hours", 0, 12, 7)

internet_hours = st.slider("Internet Hours", 0, 12, 3)

family_support = st.slider("Family Support", 0, 5, 3)

part_time_job = st.selectbox(
    "Part Time Job",
    [0, 1]
)

extracurricular = st.selectbox(
    "Extracurricular Activities",
    [0, 1]
)

parent_edu = st.slider("Parent Education Level", 0, 5, 2)

gender = st.selectbox(
    "Gender",
    [0, 1]
)

school_type = st.selectbox(
    "School Type",
    [0, 1]
)

distance_km = st.slider("Distance from School (km)", 0, 50, 5)

quiz_avg = st.slider("Quiz Average", 0, 100, 70)

midterm_score = st.slider("Midterm Score", 0, 100, 75)

# ─────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────

if st.button("Predict"):

    start = time.time()

    input_data = pd.DataFrame([{

        "study_hours": study_hours,
        "attendance_pct": attendance_pct,
        "prev_gpa": prev_gpa,
        "assignments_done": assignments_done,
        "sleep_hours": sleep_hours,
        "internet_hours": internet_hours,
        "family_support": family_support,
        "part_time_job": part_time_job,
        "extracurricular": extracurricular,
        "parent_edu": parent_edu,
        "gender": gender,
        "school_type": school_type,
        "distance_km": distance_km,
        "quiz_avg": quiz_avg,
        "midterm_score": midterm_score

    }])

    input_data = input_data[feature_columns[:-1]]

    prediction = model.predict(input_data)[0]

    elapsed = time.time() - start

    st.success(f"Predicted Score: {prediction:.2f}")

    st.info(f"Inference Time: {elapsed:.4f} sec")