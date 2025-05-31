from flask import Flask, render_template, request
import numpy as np
import pickle
import os
import pandas as pd
from datetime import datetime
app = Flask(__name__, static_url_path='/static', static_folder='static', template_folder='templates')
app.name = "Student Mental Health Prediction"


model_file = "xgb_model.pkl"  
if not os.path.exists(model_file):
    raise FileNotFoundError(f"Model file '{model_file}' not found!")

with open(model_file, "rb") as file:
    clf_model = pickle.load(file)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        time_stamp = datetime.now()
        name = request.form.get('name', 'Student')
        Gender = int(request.form.get("Gender", -1))
        Age = int(request.form.get("Age", 18))
        Academic_Pressure = float(request.form.get("Academic_Pressure", 5))
        CGPA = float(request.form.get("CGPA", 7))
        Study_Satisfaction = float(request.form.get("Study_Satisfaction", 5))
        Sleep_Duration = int(request.form.get("Sleep_Duration", 1))
        Dietary_Habits = int(request.form.get("Dietary_Habits", 1))
        Degree = int(request.form.get("Degree", -1))
        Suicidal_Thoughts = int(request.form.get("Suicidal_Thoughts", 0))
        Work_Study_Hours = float(request.form.get("Work_Study_Hours", 4))
        Financial_Stress = float(request.form.get("Financial_Stress", 5))
        Family_History_Mental_Illness = int(request.form.get("Family_History_Mental_Illness", 0))
        Save_data = int(request.form.get("Save_Data",0))

        input_data = np.array([[
            Gender, Age, Academic_Pressure, CGPA, Study_Satisfaction,
            Sleep_Duration, Dietary_Habits, Degree, Suicidal_Thoughts,
            Work_Study_Hours, Financial_Stress, Family_History_Mental_Illness
        ]])

        # Make predictions
        prediction = clf_model.predict(input_data)
        


        prediction_prob = clf_model.predict_proba(input_data) * 100

        # Interpretation of prediction
        result = "The Student's Mental Health is not good" if prediction[0] == 1 else "The Student's Mental Health is good"
        prob_good = f"{prediction_prob[0][0]:.2f}"
        prob_not_good = f"{prediction_prob[0][1]:.2f}"

        # Generate suggestions based on input values
        suggestions = []

        # Sleep Duration
        if Sleep_Duration == 0:
            suggestions.append("Try to get at least 7-8 hours of sleep for better mental and physical health.")
        elif Sleep_Duration == 2:
            suggestions.append("Less than 5 hours of sleep is harmful. Aim for at least 7 hours.")
        elif Sleep_Duration == 3:
            suggestions.append("More than 8 hours of sleep can cause lethargy. Keep it around 7-8 hours.")
        else:
            suggestions.append("Your sleep duration is good. Keep it up!")

        # Academic Pressure
        if Academic_Pressure > 7:
            suggestions.append("Consider time management techniques or seeking academic counseling to reduce stress.")

        # Dietary Habits
        if Dietary_Habits == 2:
            suggestions.append("Improve your diet by including more fruits, vegetables, and whole grains.")
        elif Dietary_Habits == 1:
            suggestions.append("Try to consume healthy food regularly and limit junk food.")

        # Study Satisfaction
        if Study_Satisfaction < 5:
            suggestions.append("Experiment with different study methods to find what works best for you.")

        # Financial Stress
        if Financial_Stress > 7:
            suggestions.append("Consider financial planning or seeking advice to manage financial stress.")

        # Suicidal Thoughts
        if Suicidal_Thoughts > 7:
            suggestions.append("Reach out to a mental health professional for support.")
        elif 1 <= Suicidal_Thoughts <= 7:
            suggestions.append("Try engaging in positive activities and spend time with loved ones.")

        # Family History of Mental Illness
        if Family_History_Mental_Illness == 1:
            suggestions.append("Stay proactive about your mental health with regular self-check-ins.")

        # General Well-being
        suggestions.append("Stay physically active to improve mood and well-being.")
        suggestions.append("Practice mindfulness or relaxation techniques to reduce stress.")

        # Convert categorical values to readable text
        gender_text = "Male" if Gender == 1 else "Female"
        sleep_text = ["5-6 hours", "7-8 hours", "Less than 5 hours", "More than 8 hours"][Sleep_Duration]
        diet_text = ["Healthy", "Moderate", "Unhealthy"][Dietary_Habits]
        degree_dict = {
            0: "B.Com", 1: "BA", 2: "BCA/BBA", 3: "BSc", 4: "Class 12",
            5: "MTech", 6: "MCA/MBA", 7: "BTech", 8: "MSc", 9: "MA", 10: "M.Com"
        }
        degree_text = degree_dict.get(Degree, "Unknown")
        suicidal_text = "Yes" if Suicidal_Thoughts == 1 else "No"
        family_history_text = "Yes" if Family_History_Mental_Illness == 1 else "No"
        if Save_data == 1:
            df = pd.DataFrame({
            'Timestamp': [time_stamp],  # Fixed column name
            'Name': [name],
            'Age': [Age],
            'Gender': [gender_text],
            'Academic Pressure': [Academic_Pressure],
            'CGPA': [CGPA],
            'Study Satisfaction': [Study_Satisfaction],  # Fixed column name
            'Sleep Duration': [sleep_text],
            'Dietary Habits': [diet_text],
            'Degree': [degree_text],
            'Suicidal Thoughts': [suicidal_text],  # Fixed column name
            'Work Study Hours': [Work_Study_Hours],
            'Financial Stress': [Financial_Stress],
            'Family History': [family_history_text],  # Fixed extra space issue
            'Good Mental Health Probability': [prob_good],  # Fixed name
            'Not Good Mental Health Probability': [prob_not_good]
                })
            

        # Save the data
            df.to_csv("Result.csv", mode='a', header=not os.path.exists("Result.csv"), index=False)

        # Save confirmation message
            save_message = "Thanks for allowing us to save your data."
        else:
            save_message = "You did not opt to save data."

   # Render results page
        return render_template(
            'result.html',
            result=result,
            suggestions=suggestions,
            prob_good=prob_good,
            prob_not_good=prob_not_good,
            name=name,
            Age=Age,
            Gender=gender_text,
            Academic_Pressure=Academic_Pressure,
            CGPA=CGPA,
            Study_Satisfaction=Study_Satisfaction,
            Sleep_Duration=sleep_text,
            Dietary_Habits=diet_text,
            Degree=degree_text,
            Suicidal_Thoughts=suicidal_text,
            Work_Study_Hours=Work_Study_Hours,
            Financial_Stress=Financial_Stress,
            Family_History_Mental_Illness=family_history_text,
            save_message = save_message
        )

    except Exception as e:
        return f"An error occurred: {e}"


if __name__ == '__main__':
    app.run(debug  = True)

