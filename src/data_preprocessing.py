# data_preprocessing.py

# Import necessary libraries
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset
def load_data():
    iris = datasets.load_iris()
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    iris_df['species'] = iris.target_names[iris.target]
    return iris_df

# Explore the dataset
def explore_data(df):
    print("Shape of the dataset:", df.shape)
    print("\nMissing values:\n", df.isnull().sum())
    print("\nData types:\n", df.dtypes)
    print("\nSpecies distribution:\n", df['species'].value_counts())
    print("\nSummary statistics:\n", df.describe())

    # Visualize pairplot
    sns.pairplot(df, hue='species')
    plt.show()

    # Visualize correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.drop(columns = ['species']).corr(), annot=True, cmap='coolwarm')
    plt.title('Feature Correlation')
    plt.show()

# Split the dataset into training and testing sets
def split_data(df):
    X = df[datasets.load_iris().feature_names]
    y = df['species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Load and explore the dataset
    iris_df = load_data()
    explore_data(iris_df)

    # Split the data
    X_train, X_test, y_train, y_test = split_data(iris_df)
    print("\nTraining and Testing sets created.")
    print("X_train shape:", X_train.shape, " | X_test shape:", X_test.shape)
