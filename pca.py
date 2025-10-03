# -------------------------------------------------------------------------
# AUTHOR: Timothy Tsang
# FILENAME: pca.py
# SPECIFICATION: description of the program
# FOR: CS 4440 (Data Mining) - Assignment #2
# TIME SPENT: 2 hours
# -----------------------------------------------------------*/

#importing some Python libraries
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#Load the data
#--> add your Python code here
df =  pd.read_csv('heart_disease_dataset.csv')

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

pc1_variances = {}

#Get the number of features
#--> add your Python code here
num_features = df.shape[1]

# Run PCA using 9 features, removing one feature at each iteration
for i in range(num_features):
    # Create a new dataset by dropping the i-th feature
    # --> add your Python code here
    removed_feature = df.columns[i]
    reduced_data = df.drop(columns=[removed_feature])

    # Run PCA on the reduced dataset
    # --> add your Python code here
    pca = PCA(n_components=1)
    pca.fit(reduced_data)

    #Store PC1 variance and the feature removed
    #Use pca.explained_variance_ratio_[0] and df_features.columns[i] for that
    # --> add your Python code here
    pc1_variance = pca.explained_variance_ratio_[0]
    pc1_variances[removed_feature] = pc1_variance

# Find the maximum PC1 variance
# --> add your Python code here
    max_variance_feature = max(pc1_variances, key = pc1_variances.get)
    max_variance_value = pc1_variances[max_variance_feature]

#Print results
#Use the format: Highest PC1 variance found: ? when removing ?
for feature, feature_val in pc1_variances.items():
    print(f"{feature}: {feature_val:.5f}")
print(f"Highest PC1 variance found: {max_variance_value: .3f} when removing {max_variance_feature}")





