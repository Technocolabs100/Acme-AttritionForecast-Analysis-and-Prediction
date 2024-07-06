import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("C:\\Users\\HP\\Downloads\\WA_Fn-UseC_-HR-Employee-Attrition.csv")

#describe data (mean,min,max,std....,etc) & display some rows (first-last) & checking for missing values
print(df.head())
print(df.tail())
print(df.describe())
missing_values = df.isnull().sum()
print(missing_values)
##########################################################################################

#showing the distribution of some of the numerical data
df[['Age', 'DailyRate', 'HourlyRate', 'JobSatisfaction', 'MonthlyIncome', 'OverTime', 'YearsSinceLastPromotion']].hist()
plt.show()

#showing the distribution of some of the categorical data
sns.countplot(data=df, x='BusinessTravel').set_title('Business Travel Distribution')
plt.show()

sns.countplot(data=df, x='Gender').set_title('Gender Distribution')
plt.show()
##########################################################################################

#displaying relation between some variables (with each other & target variable)
plt.figure(figsize=(10, 6))
plt.scatter(df['TotalWorkingYears'], df['MonthlyIncome'])
plt.title('Relationship between Total Working Years and Monthly Income')
plt.xlabel('Total Working Years')
plt.ylabel('Monthly Income')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='Attrition', hue='Gender', data=df)
plt.title('Relationship between Attrition and Gender')
plt.xlabel('Attrition')
plt.ylabel('Gender')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='Attrition', hue='MaritalStatus', data=df)
plt.title('Relationship between Attrition and Marital Status')
plt.xlabel('Attrition')
plt.ylabel('Marital Status')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='Attrition', hue='OverTime', data=df)
plt.title('Relationship between Attrition and Over Time')
plt.xlabel('Attrition')
plt.ylabel('Over time')
plt.show()
##########################################################################################

#check certain columns for non-variance
is_same_employeeCount = df['EmployeeCount'].nunique() == 1
print(is_same_employeeCount)

is_same_over18 = df['Over18'].nunique() == 1
print(is_same_over18)

is_same_StandardHours = df['StandardHours'].nunique() == 1
print(is_same_StandardHours)

df = df.drop(['EmployeeCount', 'Over18', 'StandardHours'], axis=1)
##########################################################################################

#encoding numerical & categorical variables
label_mapping = {'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2}
df['BusinessTravel'] = df['BusinessTravel'].map(label_mapping)
df['Attrition'] = pd.get_dummies(df['Attrition'], drop_first=True).astype(int)
df['Gender'] = pd.get_dummies(df['Gender'], drop_first=True).astype(int)
df['OverTime'] = pd.get_dummies(df['OverTime'], drop_first=True).astype(int)
df = pd.get_dummies(df, columns=['MaritalStatus'])
df = pd.get_dummies(df, columns=['Department'])
df = pd.get_dummies(df, columns=['EducationField'])
df = pd.get_dummies(df, columns=['JobRole'])
##########################################################################################

#scalling numerical variables
scaler = StandardScaler()
df[['DailyRate', 'HourlyRate', 'MonthlyIncome', 'MonthlyRate']] = scaler.fit_transform(df[['DailyRate', 'HourlyRate', 'MonthlyIncome', 'MonthlyRate']])

#splitting data into training and testing & creating ml
X = df.drop('Attrition', axis=1)
y = df['Attrition']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#Generating accuracy, MSE, classification report
print('Accuracy:', accuracy_score(y_test, y_pred))

mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)

print(classification_report(y_test, y_pred))