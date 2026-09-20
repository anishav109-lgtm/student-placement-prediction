from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# 1. Train the Machine Learning Model
np.random.seed(42)
n_samples = 500

cgpa = np.random.uniform(5.0, 10.0, n_samples)
tech_skills = np.random.randint(1, 6, n_samples)
comm_skills = np.random.randint(1, 4, n_samples)
internship = np.random.choice([0, 1], n_samples)

placed = ((cgpa > 7.0) & (tech_skills >= 3) & (comm_skills >= 2)).astype(int)

X = pd.DataFrame({
    'cgpa': cgpa,
    'tech_skills': tech_skills,
    'comm_skills': comm_skills,
    'internship': internship
})
y = placed

model = RandomForestClassifier()
model.fit(X, y)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Read values submitted from the HTML form using request.form
    cgpa_val = float(request.form.get('cgpa', 0))
    tech_val = int(request.form.get('tech_skills', 0))
    comm_val = int(request.form.get('comm_skills', 0))
    intern_val = int(request.form.get('internship', 0))

    features = np.array([[cgpa_val, tech_val, comm_val, intern_val]])
    prediction = model.predict(features)

    result = "Placed! 🎉" if prediction[0] == 1 else "Not Placed 😔"
    return render_template('index.html', prediction_text=f'Prediction: {result}')


if __name__ == '__main__':
    app.run(debug=True)