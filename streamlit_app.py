import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
@st.cache_data
def load_data():
    csv_file_path = r"C:\Users\Samson\Desktop\Diabetes Prediction\diabetes.csv"

    df = pd.read_csv(csv_file_path)
    return df

# Load the data
df = load_data()

# Display the first few rows of the dataset
st.write("### Dataset Preview")
st.write(df.head())


target_column = 'Outcome'  

# Split the data into features and target
X = df.drop(target_column, axis=1)
y = df[target_column]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a logistic regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Display the evaluation results
st.write("### Model Evaluation")
st.write(f"Accuracy: {accuracy}")
st.write("Classification Report:")
st.write(report)

# User input for prediction
st.write("### Make a Prediction")
input_data = {}
for column in X.columns:
    input_data[column] = st.number_input(f"Enter {column}", value=0.0)

# Convert input data to DataFrame
input_df = pd.DataFrame([input_data])

# Standardize the input data
input_df_scaled = scaler.transform(input_df)

# Make a prediction
prediction = model.predict(input_df_scaled)

# Display the prediction
st.write("### Diabetes Prediction Result")
st.write(f"Predicted Class: {prediction[0]}")
