import numpy as np
import joblib

# Load the trained K-nearest neighbors model
model = joblib.load('iris_classification_model.pkl')

# Define the target names (class labels)
target_names = ['setosa', 'versicolor', 'virginica']

def predict_flower_class(sepal_length, sepal_width, petal_length, petal_width):
    try:
        # Make predictions using the loaded model
        input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        prediction = model.predict(input_features)[0]

        # Get the corresponding target name (class label)
        target_name = target_names[prediction]

        return target_name
    except Exception as e:
        return f'Error: {str(e)}'


    # Get user input
sepal_length = float(input("Enter sepal length: "))
sepal_width = float(input("Enter sepal width: "))
petal_length = float(input("Enter petal length: "))
petal_width = float(input("Enter petal width: "))

    # Predict the flower class
prediction = predict_flower_class(sepal_length, sepal_width, petal_length, petal_width)

print(f"The predicted flower class is: {prediction}")
