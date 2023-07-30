import numpy as np
import joblib

# Load the trained linear regression model
model = joblib.load('salary_prediction_model.pkl')

def predict_salary(age, experience):
    try:
        # Make predictions using the loaded model
        input_features = np.array([[age, experience]])

        prediction = model.predict(input_features)[0]

        return prediction
    except Exception as e:
        return f'Error: {str(e)}'

if __name__ == '__main__':
    # Get user input
    age = float(input("Enter your age: "))
    experience = float(input("Enter your years of experience: "))

    # Predict the salary
    prediction = predict_salary(age, experience)

    print(f"Your predicted salary is: {prediction}")
