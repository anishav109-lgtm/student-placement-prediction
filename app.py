from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Create sample dataset
np.random.seed(42)
n_samples = 500

cgpa = np.random.uniform(5.0, 10.0, n_samples)
tech_skills = np.random.randint(1, 6, n_samples)
comm_skills = np.random.randint(1, 6, n_samples)
internship = np.random.randint(0, 2, n_samples)

# Placement target
placed = (
    (cgpa >= 7.0) &
    (tech_skills >= 3) &
    (comm_skills >= 3) &
    (internship == 1)
).astype(int)

# Prepare dataset
X = pd.DataFrame({
    "cgpa": cgpa,
    "tech_skills": tech_skills,
    "comm_skills": comm_skills,
    "internship": internship
})

y = placed

# Train Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)


# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Prediction
@app.route("/predict", methods=["POST"])
def predict():

    try:
        cgpa_val = float(request.form["cgpa"])
        tech_val = int(request.form["tech_skills"])
        comm_val = int(request.form["comm_skills"])
        intern_val = int(request.form["internship"])

        # Create input data
        input_data = pd.DataFrame([{
            "cgpa": cgpa_val,
            "tech_skills": tech_val,
            "comm_skills": comm_val,
            "internship": intern_val
        }])

        # Make prediction
        prediction = model.predict(input_data)[0]

        if prediction == 1:
            result = "Placed"
        else:
            result = "Not Placed"

        return render_template(
            "index.html",
            prediction_text=f"Prediction: {result}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}"
        )


# Run application
if __name__ == "__main__":
    app.run(debug=True)
