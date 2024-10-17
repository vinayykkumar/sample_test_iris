from sklearn import datasets
import pandas as pd

# Load the iris dataset
iris = datasets.load_iris()
# Convert it to a DataFrame for easier manipulation
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target_names[iris.target]

# Display the first few rows of the dataset
print(iris_df.head())
