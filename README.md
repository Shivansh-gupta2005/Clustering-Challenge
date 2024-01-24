# Clustering-Challenge
SPRING CAMP RECRUITMENT TASK
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import gradio as gr
# Load the dataset
url = "link_to_kaggle_dataset"
df = pd.read_csv(url)

# Explore the dataset
print(df.head())
print(df.info())
# Handle missing values, categorical variables, and feature scaling if needed
# For simplicity, we assume the dataset is clean and numerical
# Separate features and target variable
X = df.drop('SalePrice', axis=1)
y = df['SalePrice']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize KNN model and fit on the training data
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Evaluate the model
accuracy = knn_model.score(X_test, y_test)
print(f"KNN Model Accuracy: {accuracy}")
# Use k-means clustering to gain insights into factors associated with houses of similar prices
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

# Visualize clustering
plt.scatter(df['Feature1'], df['Feature2'], c=df['cluster'], cmap='viridis')
plt.title('Clustering of Houses based on Features')
plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.show()
# Define the Gradio interface for house price prediction
def predict_price(features):
    features = np.array(features).reshape(1, -1)
    prediction = knn_model.predict(features)[0]
    return f'Predicted House Price: ${prediction:.2f}'

iface = gr.Interface(fn=predict_price, inputs="text", outputs="text")
iface.launch(share=True)
