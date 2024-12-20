import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore')
from sklearn.preprocessing import scale, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import KFold
import missingno as msno

# Load the dataset
csv_file_path = 'diabetes.csv'  # Ensure the file path is correct for your deployment environment
df = pd.read_csv(csv_file_path)

# Display the first few rows of the DataFrame
print(df.head())

df.info()

## EDA
df.describe()
df['Outcome'].value_counts() * 100 / len(df)

# Plot the hist of the age variable
plt.figure(figsize=(8, 7))
plt.xlabel("Age", fontsize=10)
plt.ylabel("Count", fontsize=10)
df['Age'].hist(edgecolor="black")
plt.show()

# Set up the subplots
fig, ax = plt.subplots(4, 2, figsize=(20, 20))

# Plotting each variable on the corresponding axis
sns.histplot(df.Pregnancies, bins=20, ax=ax[0, 0], color='red', kde=True)
sns.histplot(df.Glucose, bins=20, ax=ax[0, 1], color='red', kde=True)
sns.histplot(df.BloodPressure, bins=20, ax=ax[1, 0], color='red', kde=True)
sns.histplot(df.SkinThickness, bins=20, ax=ax[1, 1], color='red', kde=True)
sns.histplot(df.Insulin, bins=20, ax=ax[2, 0], color='red', kde=True)
sns.histplot(df.BMI, bins=20, ax=ax[2, 1], color='red', kde=True)
sns.histplot(df.DiabetesPedigreeFunction, bins=20, ax=ax[3, 0], color='red', kde=True)
sns.histplot(df.Age, bins=20, ax=ax[3, 1], color='red', kde=True)

# Display the plot
plt.tight_layout()
plt.show()

df.groupby("Outcome").agg({"Pregnancies": "mean"})
df.groupby("Outcome").agg({"Pregnancies": "max"})
df.groupby("Outcome").agg({"Glucose": "mean"})
df.groupby("Outcome").agg({"Glucose": "max"})
df.groupby("Outcome").agg({"BloodPressure": "mean"})
df.groupby("Outcome").agg({"BloodPressure": "max"})
df.groupby("Outcome").agg({"SkinThickness": "mean"})
df.groupby("Outcome").agg({"SkinThickness": "max"})
df.groupby("Outcome").agg({"Insulin": "mean"})
df.groupby("Outcome").agg({"Insulin": "max"})
df.groupby("Outcome").agg({"BMI": "mean"})
df.groupby("Outcome").agg({"BMI": "max"})
df.groupby("Outcome").agg({"DiabetesPedigreeFunction": "mean"})
df.groupby("Outcome").agg({"DiabetesPedigreeFunction": "max"})
df.groupby("Outcome").agg({"Age": "mean"})
df.groupby("Outcome").agg({"Age": "max"})

# Create subplots
f, ax = plt.subplots(1, 2, figsize=(18, 8))

# Pie chart
df['Outcome'].value_counts().plot.pie(
    explode=[0, 0.1],
    autopct="%1.1f%%",
    ax=ax[0],
    shadow=True
)
ax[0].set_title('Target Distribution')
ax[0].set_ylabel('')

# Count plot
sns.countplot(x="Outcome", data=df, ax=ax[1])
ax[1].set_title("Outcome")

# Show plot
plt.tight_layout()
plt.show()

f, ax = plt.subplots(figsize=[20, 15])
sns.heatmap(df.corr(), annot=True, fmt='.2f', ax=ax, cmap='magma')
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

# Data Preprocessing
df.isnull().sum()

df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
    'BMI', 'DiabetesPedigreeFunction', 'Age']] = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
    'BMI', 'DiabetesPedigreeFunction', 'Age']].replace(0, np.NaN)

df.isnull().sum()
msno.bar(df, color="red")
plt.show()

# Median
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

df.isnull().sum()

# Outlier detection
for feature in df:
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    if df[(df[feature] > upper)].any(axis=None):
        print(feature, 'yes')
    else:
        print(feature, 'no')

plt.figure(figsize=(8, 7))
sns.boxplot(x=df['Insulin'], color="red")
plt.show()

Q1 = df.Insulin.quantile(0.25)
Q3 = df.Insulin.quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
df.loc[df['Insulin'] > upper, 'Insulin'] = upper

plt.figure(figsize=(8, 7))
sns.boxplot(x=df['Insulin'], color="red")
plt.show()

# LOF
lof = LocalOutlierFactor(n_neighbors=10)
lof.fit_predict(df)

plt.figure(figsize=(8, 7))
sns.boxplot(x=df['Pregnancies'], color="red")
plt.show()

df_scores = lof.negative_outlier_factor_
np.sort(df_scores)[0:20]

threshold = np.sort(df_scores)[7]
print(threshold)

outlier = df_scores > threshold
df = df[outlier]
print(df)

### Feature Engineering
NewBMI = pd.Series(['Underweight', 'Normal', 'Overweight', 'Obesity 1', 'Obesity 2', 'Obesity 3'], dtype='category')
print(NewBMI)

df['NewBMI'] = NewBMI
df.loc[df['BMI'] < 18.5, "NewBMI"] = NewBMI[0]
df.loc[(df['BMI'] > 18.5) & (df['BMI'] <= 24.9), 'NewBMI'] = NewBMI[1]
df.loc[(df['BMI'] > 24.9) & (df['BMI'] <= 29.9), 'NewBMI'] = NewBMI[2]
df.loc[(df['BMI'] > 29.9) & (df['BMI'] <= 34.9), 'NewBMI'] = NewBMI[3]
df.loc[(df['BMI'] > 34.9) & (df['BMI'] <= 39.9), 'NewBMI'] = NewBMI[4]
df.loc[df['BMI'] > 39.9, 'NewBMI'] = NewBMI[5]
print(df.head())

# If insulin > 16 & insulin < 166 = normal
def set_insulin(row):
    if row['Insulin'] > 16 and row['Insulin'] <= 166:
        return 'Normal'
    else:
        return 'Abnormal'

df = df.assign(NewInsulinScore=df.apply(set_insulin, axis=1))
df['Glucose'].fillna(df['Glucose'].median(), inplace=True)
print(df.head())

### Feature Engineering
# Check for any non-numeric values that were coerced to NaN
if df['Glucose'].isnull().any():
    print("Warning: Some Glucose values could not be converted to numeric and were set to NaN.")
# Initialize the NewGlucose column
df['NewGlucose'] = None
df['Glucose'] = pd.to_numeric(df['Glucose'], errors='coerce')
NewGlucose = pd.Series(['Low', 'Normal', 'Overweight', 'Secret', 'High'], dtype='category')

df['NewGlucose'] = NewGlucose
df.loc[df['Glucose'] <= 70, 'NewGlucose'] = NewGlucose[0]
df.loc[(df['Glucose'] > 70) & (df['Glucose'] <= 99), 'NewGlucose'] = NewGlucose[1]
df.loc[(df['Glucose'] > 99) & (df['Glucose'] <= 126), 'NewGlucose'] = NewGlucose[2]
df.loc[df['Glucose'] > 126, 'NewGlucose'] = NewGlucose[3]

## One hot encoding: To handle the categorical values
df = pd.get_dummies(df, columns=['NewBMI', 'NewInsulinScore', 'NewGlucose'], drop_first=True)

# Convert True/False to 1/0
bool_columns = df.select_dtypes(include='bool').columns
df[bool_columns] = df[bool_columns].astype(int)

categorical_df = df[['NewBMI_Obesity 1',
    'NewBMI_Obesity 2', 'NewBMI_Obesity 3', 'NewBMI_Overweight',
    'NewBMI_Underweight', 'NewInsulinScore_Normal', 'NewGlucose_Low',
    'NewGlucose_Normal', 'NewGlucose_Overweight', 'NewGlucose_Secret']]
print(categorical_df)

y = df['Outcome']
X = df.drop(['Outcome', 'NewBMI_Obesity 1',
    'NewBMI_Obesity 2', 'NewBMI_Obesity 3', 'NewBMI_Overweight',
    'NewBMI_Underweight', 'NewInsulinScore_Normal', 'NewGlucose_Low',
    'NewGlucose_Normal', 'NewGlucose_Overweight', 'NewGlucose_Secret'], axis=1)

# Access columns and index
cols = X.columns
index = X.index

from sklearn.preprocessing import RobustScaler
transformer = RobustScaler().fit(X)
X = transformer.transform(X)
X = pd.DataFrame(X, columns=cols, index=index)
print(X.head())

X = pd.concat([X, categorical_df], axis=1)
X.head()

## Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Training
## Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
accuracy_score(y_train, log_reg.predict(X_train))
accuracy_score(y_test, log_reg.predict(X_test))
log_reg_acc = accuracy_score(y_test, log_reg.predict(X_test))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

## KNN
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
knn_acc = accuracy_score(y_test, knn.predict(X_test))
print(accuracy_score(y_train, knn.predict(X_train)))
print(accuracy_score(y_test, knn.predict(X_test)))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

## SVM
# Hyper parameter tuning
svc = SVC(probability=True)
parameter = {
    "gamma": [0.0001, 0.001, 0.01, 0.1],
    "C": [0.01, 0.05, 0.5, 0.01, 1, 10, 15, 20]
}
grid_search = GridSearchCV(svc, parameter)
grid_search.fit(X_train, y_train)

# best_parameter
print(grid_search.best_params_)
print(grid_search.best_score_)

svc = SVC(C=10, gamma=0.1, probability=True)
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
svc_acc = accuracy_score(y_test, svc.predict(X_test))
print(accuracy_score(y_train, svc.predict(X_train)))
print(accuracy_score(y_test, svc.predict(X_test)))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

## Decision Tree
DT = DecisionTreeClassifier()
DT.fit(X_train, y_train)
y_pred = DT.predict(X_test)
print(accuracy_score(y_train, DT.predict(X_train)))
print(accuracy_score(y_test, DT.predict(X_test)))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Hyperparameter tuning of Decision Tree
grid_param = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 7, 10],
    'splitter': ['best', 'random'],
    'min_samples_leaf': [1, 2, 3, 5, 7],
    'min_samples_split': [1, 2, 3, 5, 7],
    'max_features': ['auto', 'sqrt', 'log2']
}
grid_search_dt = GridSearchCV(DT, grid_param, cv=50, n_jobs=-1, verbose=1)
grid_search_dt.fit(X_train, y_train)

print(grid_search_dt.best_params_)
print(grid_search_dt.best_score_)

DT = grid_search_dt.best_estimator_
y_pred = DT.predict(X_test)
print(accuracy_score(y_train, DT.predict(X_train)))
dt_acc = accuracy_score(y_test, DT.predict(X_test))
print(accuracy_score(y_test, DT.predict(X_test)))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Random Forest Classifier
rand_clf = RandomForestClassifier(criterion='entropy', max_depth=15, max_features=0.75, min_samples_split=3, n_estimators=130)
rand_clf.fit(X_train, y_train)

y_pred = rand_clf.predict(X_test)

y_pred = rand_clf.predict(X_test)
print(accuracy_score(y_train, rand_clf.predict(X_train)))
rand_acc = accuracy_score(y_test, rand_clf.predict(X_test))
print(accuracy_score(y_test, rand_clf.predict(X_test)))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# GradientBoosting Classifier
gbc = GradientBoostingClassifier()

parameters = {
    'loss': ['deviance', 'exponential'],
    'learning_rate': [0.001, 0.1, 1, 10],
    'n_estimators': [100, 150, 180, 200]
}
grid_search_gbc = GridSearchCV(gbc, parameters, cv=50, n_jobs=-1, verbose=1)
grid_search_gbc.fit(X_train, y_train)

print(grid_search_gbc.best_params_)
print(grid_search_gbc.best_score_)

gbc = GradientBoostingClassifier(learning_rate=0.1, loss='exponential', n_estimators=150)
gbc.fit(X_train, y_train)

gbc = grid_search_dt.best_estimator_
y_pred = gbc.predict(X_test)
print(accuracy_score(y_train, gbc.predict(X_train)))
gbc_acc = accuracy_score(y_test, gbc.predict(X_test))
print(accuracy_score(y_test, gbc.predict(X_test)))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Model Comparison
models = pd.DataFrame({
    'Model': ['Logistic Regression', 'KNN', 'SVM', 'Decision Tree Classifier', 'Random Forest Classifier', 'Gradient Boosting Clasifier'],
    'Score': [100 * round(log_reg_acc, 4), 100 * round(knn_acc, 4), 100 * round(svc_acc, 4), 100 * round(dt_acc, 4),
              100 * round(rand_acc, 4), 100 * round(gbc_acc, 4)]
})
models.sort_values(by='Score', ascending=False)
