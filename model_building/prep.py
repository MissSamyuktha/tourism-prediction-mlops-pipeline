# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/MissSamyuktha/tourism-prediction/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop the unique identifier and redundant columns
df.drop(columns=['CustomerID', 'Unnamed: 0'], inplace=True)

# Correcting Data entry inconsistency in Gender column
df['Gender'] = df['Gender'].replace({'Fe Male': 'Female'})

# Define the target variable for the classification task
target = 'ProdTaken'

# List of numerical features in the dataset
numeric_features = [
    'Age', 'DurationOfPitch', 'MonthlyIncome', # continuos features
    'CityTier', 'NumberOfPersonVisiting', 'NumberOfFollowups', 
    'PreferredPropertyStar', 'NumberOfTrips', 'Passport',
    'PitchSatisfactionScore', 'OwnCar', 'NumberOfChildrenVisiting' # discrete features
]
    
# List of categorical features in the dataset
categorical_features = [
    'TypeofContact', 'Occupation', 'Gender', 
    'ProductPitched', 'MaritalStatus', 'Designation'         
]

# Define predictor matrix (X) using selected numeric and categorical features
X = df[numeric_features + categorical_features]

# Define target variable
y = df[target]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="MissSamyuktha/tourism-prediction",
        repo_type="dataset",
    )
