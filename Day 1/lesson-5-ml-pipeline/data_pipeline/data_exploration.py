import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
columns = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race",
    "sex", "capital-gain", "capital-loss", "hours-per-week",
    "native-country", "income"
]

data = pd.read_csv(r"/Users/aelkhodary/Documents/GitHub/oreilly-mlops-bootcamp/Day 1/lesson-5-ml-pipeline/data/adult.data", header=None, names=columns, na_values=" ?")
print("data shape : ", data.shape)
print(data.head(5))

# Dataset info
print("\nDataset Information:")
print(data.info())

# Check for missing values
print("\nMissing Values in Each Column:")
print(data.isnull().sum())

# Summary statistics
print("\nSummary Statistics:")
print(data.describe(include="all"))

# Check class distribution
print("\nClass Distribution:")
print(data["income"].value_counts())

# # Visualize class distribution
# sns.countplot(data=data, x="income")
# plt.title("Class Distribution")
# plt.show()
