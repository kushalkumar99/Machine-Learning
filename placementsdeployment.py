import numpy as np
import joblib

# Load the trained logistic regression model
model = joblib.load('placements_prediction_model.pkl')

def predict_placement_chances(cgpa):
    try:
        # Make predictions using the loaded model
        input_features = np.array([[cgpa]])

        prediction = model.predict(input_features)[0]

        # Get the corresponding target name (class label)
        if prediction == 1:
            target_name = "Placed"
        else:
            target_name = "Not placed"

        return target_name
    except Exception as e:
        return f'Error: {str(e)}'

if __name__ == '__main__':
    # Get user input
    cgpa = float(input("Enter your CGPA: "))

    # Predict the placement chances
    prediction = predict_placement_chances(cgpa)

    print(f"Your chances of getting placed are: {prediction}")
