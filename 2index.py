# dataset_generator.py

import numpy as np
import pandas as pd

np.random.seed(42)

# 500 students ka data
n = 500

study_hours = np.random.uniform(1, 10, n)
attendance = np.random.uniform(50, 100, n)
previous_grades = np.random.uniform(40, 95, n)

# Final marks correlated with features
final_marks = (
    5 * study_hours +
    0.3 * attendance +
    0.5 * previous_grades +
    np.random.normal(0, 5, n)
)

data = pd.DataFrame({
    "Study_Hours": study_hours,
    "Attendance": attendance,
    "Previous_Grades": previous_grades,
    "Final_Marks": final_marks
})

# Random missing values add karte hain
for col in data.columns:
    data.loc[data.sample(frac=0.02).index, col] = np.nan

data.to_csv("student_data.csv", index=False)

print("Dataset created successfully!")
