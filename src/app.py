# app.py

from flask import Flask, request, render_template, jsonify
import pickle
from data_preprocessing import load_data, split_data
from model_training import train_model

app = Flask(__name__)

# Train and save the model
def train_and_save_model():
    iris_df = load_data()
    X_train, X_test, y_train, y_test = split_data(iris_df)
    model = train_model(X_train, y_train)
    with open('iris_model.pkl', 'wb') as f:
        pickle.dump(model, f)

# Load the trained model
def load_model():
    with open('iris_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for handling predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input from the form
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])

    # Prepare the input data
    input_features = [sepal_length, sepal_width, petal_length, petal_width]

    # Load the model
    model = load_model()

    # Make a prediction
    prediction = model.predict([input_features])[0]

    # Return the result in the rendered HTML
    return render_template('index.html', prediction=prediction)

if __name__ == "__main__":
    train_and_save_model()
    app.run(debug=True)
