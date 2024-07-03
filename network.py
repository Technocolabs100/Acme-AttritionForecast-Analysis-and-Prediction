import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
file_path = 'Attrition Dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv'
data = pd.read_csv(file_path)

# Display basic information about the dataset
data_info = data.info()
data_description = data.describe(include='all')
data_head = data.head()

# Check for duplicate rows
duplicates = data.duplicated().sum()

# List of numerical columns
numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns

# Plot boxplots for numerical columns to identify outliers
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(6, 6, i)
    sns.boxplot(data[col])
    plt.title(col)

plt.tight_layout()
plt.savefig('results/boxplots.png')

# Ensure 'Attrition' is present and convert it to numerical
if 'Attrition' in data.columns:
    data['Attrition'] = data['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)

# List of categorical columns excluding 'Attrition'
categorical_cols = data.select_dtypes(include=['object']).columns
categorical_cols = [col for col in categorical_cols if col != 'Attrition']

# Apply one-hot encoding to categorical columns
data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# Save the preprocessed data to a new file
data_encoded.to_csv('results/preprocessed_data.csv', index=False)

# Classification: Logistic Regression
X = data_encoded.drop('Attrition', axis=1)
y = data_encoded['Attrition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)

log_reg_results = {
    "Confusion Matrix": confusion_matrix(y_test, y_pred),
    "Classification Report": classification_report(y_test, y_pred),
    "Accuracy": accuracy_score(y_test, y_pred)
}

# Classification: Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

y_pred_rf = rf_clf.predict(X_test)

rf_clf_results = {
    "Confusion Matrix": confusion_matrix(y_test, y_pred_rf),
    "Classification Report": classification_report(y_test, y_pred_rf),
    "Accuracy": accuracy_score(y_test, y_pred_rf)
}

# Regression: Linear Regression
X_reg = data_encoded.drop('DailyRate', axis=1)
y_reg = data_encoded['DailyRate']

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

lin_reg = LinearRegression()
lin_reg.fit(X_train_reg, y_train_reg)

y_pred_reg = lin_reg.predict(X_test_reg)

lin_reg_results = {
    "Mean Absolute Error": mean_absolute_error(y_test_reg, y_pred_reg),
    "Mean Squared Error": mean_squared_error(y_test_reg, y_pred_reg),
    "R-squared": r2_score(y_test_reg, y_pred_reg)
}

# Regression: Random Forest Regressor
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train_reg, y_train_reg)

y_pred_rf_reg = rf_reg.predict(X_test_reg)

rf_reg_results = {
    "Mean Absolute Error": mean_absolute_error(y_test_reg, y_pred_rf_reg),
    "Mean Squared Error": mean_squared_error(y_test_reg, y_pred_rf_reg),
    "R-squared": r2_score(y_test_reg, y_pred_rf_reg)
}

# Create results directory if it doesn't exist
results_dir = 'results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Save key insights and model performance metrics to a text file
results_file_path = os.path.join(results_dir, 'results.txt')

with open(results_file_path, 'w') as file:
    file.write("### Dataset Information ###\n")
    file.write(f"Number of rows: {data.shape[0]}\n")
    file.write(f"Number of columns: {data.shape[1]}\n\n")
    
    file.write("### Duplicate Rows ###\n")
    file.write(f"Number of duplicate rows: {duplicates}\n\n")
    
    file.write("### Logistic Regression Results ###\n")
    file.write(f"Confusion Matrix:\n{log_reg_results['Confusion Matrix']}\n")
    file.write(f"Classification Report:\n{log_reg_results['Classification Report']}\n")
    file.write(f"Accuracy: {log_reg_results['Accuracy']}\n\n")
    
    file.write("### Random Forest Classifier Results ###\n")
    file.write(f"Confusion Matrix:\n{rf_clf_results['Confusion Matrix']}\n")
    file.write(f"Classification Report:\n{rf_clf_results['Classification Report']}\n")
    file.write(f"Accuracy: {rf_clf_results['Accuracy']}\n\n")
    
    file.write("### Linear Regression Results ###\n")
    file.write(f"Mean Absolute Error: {lin_reg_results['Mean Absolute Error']}\n")
    file.write(f"Mean Squared Error: {lin_reg_results['Mean Squared Error']}\n")
    file.write(f"R-squared: {lin_reg_results['R-squared']}\n\n")
    
    file.write("### Random Forest Regressor Results ###\n")
    file.write(f"Mean Absolute Error: {rf_reg_results['Mean Absolute Error']}\n")
    file.write(f"Mean Squared Error: {rf_reg_results['Mean Squared Error']}\n")
    file.write(f"R-squared: {rf_reg_results['R-squared']}\n")

print("Results have been saved in the 'results/' directory.")
