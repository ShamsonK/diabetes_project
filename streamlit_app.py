import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import missingno as msno

# Load the dataset
csv_file_path = 'diabetes.csv'  # Ensure the file path is correct for your deployment environment
df = pd.read_csv(csv_file_path)

# Display the first few rows of the DataFrame
st.write("First few rows of the dataset:")
st.write(df.head())

# Data Preprocessing
df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
    'BMI', 'DiabetesPedigreeFunction', 'Age']] = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
    'BMI', 'DiabetesPedigreeFunction', 'Age']].replace(0, np.NaN)

# Median imputation
def median_target(var):
    temp = df[df[var].notnull()]
    temp = temp[[var, "Outcome"]].groupby(['Outcome'])[[var]].median().reset_index()
    return temp

columns = df.columns
columns = columns.drop("Outcome")
for i in columns:
    median_target(i)
    df.loc[(df['Outcome'] == 0) & (df[i].isnull()), i] = median_target(i)[i][0]
    df.loc[(df['Outcome'] == 1) & (df[i].isnull()), i] = median_target(i)[i][1]

# Outlier detection and handling
for feature in df:
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df.loc[df[feature] > upper, feature] = upper

# Feature Engineering
NewBMI = pd.Series(['Underweight', 'Normal', 'Overweight', 'Obesity 1', 'Obesity 2', 'Obesity 3'], dtype='category')
df['NewBMI'] = NewBMI
df.loc[df['BMI'] < 18.5, "NewBMI"] = NewBMI[0]
df.loc[(df['BMI'] > 18.5) & (df['BMI'] <= 24.9), 'NewBMI'] = NewBMI[1]
df.loc[(df['BMI'] > 24.9) & (df['BMI'] <= 29.9), 'NewBMI'] = NewBMI[2]
df.loc[(df['BMI'] > 29.9) & (df['BMI'] <= 34.9), 'NewBMI'] = NewBMI[3]
df.loc[(df['BMI'] > 34.9) & (df['BMI'] <= 39.9), 'NewBMI'] = NewBMI[4]
df.loc[df['BMI'] > 39.9, 'NewBMI'] = NewBMI[5]

def set_insulin(row):
    if row['Insulin'] > 16 and row['Insulin'] <= 166:
        return 'Normal'
    else:
        return 'Abnormal'

df = df.assign(NewInsulinScore=df.apply(set_insulin, axis=1))
df['Glucose'].fillna(df['Glucose'].median(), inplace=True)

NewGlucose = pd.Series(['Low', 'Normal', 'Overweight', 'Secret', 'High'], dtype='category')
df['NewGlucose'] = NewGlucose
df.loc[df['Glucose'] <= 70, 'NewGlucose'] = NewGlucose[0]
df.loc[(df['Glucose'] > 70) & (df['Glucose'] <= 99), 'NewGlucose'] = NewGlucose[1]
df.loc[(df['Glucose'] > 99) & (df['Glucose'] <= 126), 'NewGlucose'] = NewGlucose[2]
df.loc[df['Glucose'] > 126, 'NewGlucose'] = NewGlucose[3]

# One hot encoding
df = pd.get_dummies(df, columns=['NewBMI', 'NewInsulinScore', 'NewGlucose'], drop_first=True)

# Convert True/False to 1/0
bool_columns = df.select_dtypes(include='bool').columns
df[bool_columns] = df[bool_columns].astype(int)

y = df['Outcome']
X = df.drop(['Outcome'], axis=1)

# Standard scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Model Training
## Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
log_reg_acc = accuracy_score(y_test, y_pred)

## KNN
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
knn_acc = accuracy_score(y_test, y_pred)

## SVM
svc = SVC(probability=True)
parameter = {
    "gamma": [0.0001, 0.001, 0.01, 0.1],
    "C": [0.01, 0.05, 0.5, 0.01, 1, 10, 15, 20]
}
grid_search = GridSearchCV(svc, parameter)
grid_search.fit(X_train, y_train)
svc = SVC(C=grid_search.best_params_['C'], gamma=grid_search.best_params_['gamma'], probability=True)
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
svc_acc = accuracy_score(y_test, y_pred)

## Decision Tree
DT = DecisionTreeClassifier()
DT.fit(X_train, y_train)
y_pred = DT.predict(X_test)
dt_acc = accuracy_score(y_test, y_pred)

## Random Forest Classifier
rand_clf = RandomForestClassifier(criterion='entropy', max_depth=15, max_features=0.75, min_samples_split=3, n_estimators=130)
rand_clf.fit(X_train, y_train)
y_pred = rand_clf.predict(X_test)
rand_acc = accuracy_score(y_test, y_pred)

## Gradient Boosting Classifier
gbc = GradientBoostingClassifier()
parameters = {
    'loss': ['deviance', 'exponential'],
    'learning_rate': [0.001, 0.1, 1, 10],
    'n_estimators': [100, 150, 180, 200]
}
grid_search_gbc = GridSearchCV(gbc, parameters, cv=50, n_jobs=-1, verbose=1)
grid_search_gbc.fit(X_train, y_train)
gbc = GradientBoostingClassifier(learning_rate=grid_search_gbc.best_params_['learning_rate'], loss=grid_search_gbc.best_params_['loss'], n_estimators=grid_search_gbc.best_params_['n_estimators'])
gbc.fit(X_train, y_train)
y_pred = gbc.predict(X_test)
gbc_acc = accuracy_score(y_test, y_pred)

# Model Comparison
models = pd.DataFrame({
    'Model': ['Logistic Regression', 'KNN', 'SVM', 'Decision Tree Classifier', 'Random Forest Classifier', 'Gradient Boosting Classifier'],
    'Score': [100 * round(log_reg_acc, 4), 100 * round(knn_acc, 4), 100 * round(svc_acc, 4), 100 * round(dt_acc, 4),
              100 * round(rand_acc, 4), 100 * round(gbc_acc, 4)]
})
models.sort_values(by='Score', ascending=False, inplace=True)

# Display results
st.write("Model Comparison:")
st.write(models)

# Display classification report for the best model
best_model = models.iloc[0]['Model']
if best_model == 'Logistic Regression':
    y_pred = log_reg.predict(X_test)
elif best_model == 'KNN':
    y_pred = knn.predict(X_test)
elif best_model == 'SVM':
    y_pred = svc.predict(X_test)
elif best_model == 'Decision Tree Classifier':
    y_pred = DT.predict(X_test)
elif best_model == 'Random Forest Classifier':
    y_pred = rand_clf.predict(X_test)
elif best_model == 'Gradient Boosting Classifier':
    y_pred = gbc.predict(X_test)

st.write(f"Classification Report for {best_model}:")
st.write(classification_report(y_test, y_pred))
