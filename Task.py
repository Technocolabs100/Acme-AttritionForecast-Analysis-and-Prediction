import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Load and inspect dataset
file_path = 'Attrition Dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv'
data = pd.read_csv(file_path)
data_info = data.info()
data_description = data.describe(include='all')
data_head = data.head()

# Check for duplicates
duplicate_count = data.duplicated().sum()

# Visualize numerical data to find outliers
num_cols = data.select_dtypes(include=['int64', 'float64']).columns
plt.figure(figsize=(15, 10))
for i, column in enumerate(num_cols, 1):
    plt.subplot(6, 6, i)
    sns.boxplot(data[column])
    plt.title(column)

plt.tight_layout()
plt.savefig('results/boxplots.png')

# Ensure 'Attrition' is present and convert to numerical
if 'Attrition' in data.columns:
    data['Attrition'] = data['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)

# One-hot encode categorical variables
cat_cols = data.select_dtypes(include=['object']).columns
cat_cols = [col for col in cat_cols if col != 'Attrition']
data_encoded = pd.get_dummies(data, columns=cat_cols, drop_first=True)
data_encoded.to_csv('results/preprocessed_data.csv', index=False)

# Logistic Regression model
X = data_encoded.drop('Attrition', axis=1)
y = data_encoded['Attrition']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
log_reg_metrics = {
    "Confusion Matrix": confusion_matrix(y_test, y_pred),
    "Classification Report": classification_report(y_test, y_pred),
    "Accuracy": accuracy_score(y_test, y_pred)
}

# Random Forest Classifier model
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)
rf_clf_metrics = {
    "Confusion Matrix": confusion_matrix(y_test, y_pred_rf),
    "Classification Report": classification_report(y_test, y_pred_rf),
    "Accuracy": accuracy_score(y_test, y_pred_rf)
}

# Linear Regression model
X_reg = data_encoded.drop('DailyRate', axis=1)
y_reg = data_encoded['DailyRate']
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

lin_reg = LinearRegression()
lin_reg.fit(X_train_reg, y_train_reg)
y_pred_reg = lin_reg.predict(X_test_reg)
lin_reg_metrics = {
    "Mean Absolute Error": mean_absolute_error(y_test_reg, y_pred_reg),
    "Mean Squared Error": mean_squared_error(y_test_reg, y_pred_reg),
    "R-squared": r2_score(y_test_reg, y_pred_reg)
}

# Random Forest Regressor model
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train_reg, y_train_reg)
y_pred_rf_reg = rf_reg.predict(X_test_reg)
rf_reg_metrics = {
    "Mean Absolute Error": mean_absolute_error(y_test_reg, y_pred_rf_reg),
    "Mean Squared Error": mean_squared_error(y_test_reg, y_pred_rf_reg),
    "R-squared": r2_score(y_test_reg, y_pred_rf_reg)
}

# Create results directory
results_dir = 'results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Save results
results_file = os.path.join(results_dir, 'results.txt')
with open(results_file, 'w') as file:
    file.write("### Dataset Overview ###\n")
    file.write(f"Rows: {data.shape[0]}\n")
    file.write(f"Columns: {data.shape[1]}\n\n")
    
    file.write("### Duplicate Records ###\n")
    file.write(f"Duplicate rows: {duplicate_count}\n\n")
    
    file.write("### Logistic Regression Metrics ###\n")
    file.write(f"Confusion Matrix:\n{log_reg_metrics['Confusion Matrix']}\n")
    file.write(f"Classification Report:\n{log_reg_metrics['Classification Report']}\n")
    file.write(f"Accuracy: {log_reg_metrics['Accuracy']}\n\n")
    
    file.write("### Random Forest Classifier Metrics ###\n")
    file.write(f"Confusion Matrix:\n{rf_clf_metrics['Confusion Matrix']}\n")
    file.write(f"Classification Report:\n{rf_clf_metrics['Classification Report']}\n")
    file.write(f"Accuracy: {rf_clf_metrics['Accuracy']}\n\n")
    
    file.write("### Linear Regression Metrics ###\n")
    file.write(f"Mean Absolute Error: {lin_reg_metrics['Mean Absolute Error']}\n")
    file.write(f"Mean Squared Error: {lin_reg_metrics['Mean Squared Error']}\n")
    file.write(f"R-squared: {lin_reg_metrics['R-squared']}\n\n")
    
    file.write("### Random Forest Regressor Metrics ###\n")
    file.write(f"Mean Absolute Error: {rf_reg_metrics['Mean Absolute Error']}\n")
    file.write(f"Mean Squared Error: {rf_reg_metrics['Mean Squared Error']}\n")
    file.write(f"R-squared: {rf_reg_metrics['R-squared']}\n")

print("Results have been saved in the 'results/' directory.")
