# model_training.py

# Import necessary libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from data_preprocessing import load_data, split_data

# Train the model
def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("\nModel Accuracy:", accuracy)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

if __name__ == "__main__":
    # Load and split the dataset
    iris_df = load_data()
    X_train, X_test, y_train, y_test = split_data(iris_df)

    # Train the model
    model = train_model(X_train, y_train)
    
    # Evaluate the model
    evaluate_model(model, X_test, y_test)
