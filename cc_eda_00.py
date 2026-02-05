import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


cc = pd.read_excel('cc_default.xls', header=1)
# print(cc.head())
print(cc.columns)

# Set random seed for reproducibility
random_state = 42

X = cc.drop(columns='default payment next month')
y = cc['default payment next month']

# Check for missing values
print("Missing values in each column:")
print(X.isnull().sum())

# Datatypes
print("Data types of each column:")
for col in X.columns:
    print(f"{col}: {X[col].dtype}")

# Convert variables to int if necessary
for col in X.columns:
    if X[col].dtype == 'float64':
        X[col] = X[col].astype(int)

# Check length of df
print(f"Length of DataFrame: {len(X)}")

# Check percent of positive class
positive_class_percentage = (y.sum() / len(y)) * 100
print(f"Percentage of positive class (default payment next month): {positive_class_percentage:.2f}%")

# Check correlation matrix visualization

plt.figure(figsize=(12, 10))
correlation_matrix = X.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Means of each column for positive class
for col in X.columns:
    mean_value = X[y == 1][col].mean()
    print(f"Mean value in {col} for positive class: {mean_value}")

# Mean values of each column for entire dataset
mean_values = X.mean()
print("Mean values of each column:")
print(mean_values)

# Average values of three columns for positive class and entire dataset
# BILL_AMT1 to BILL_AMT6
all_bill_cols = [f'BILL_AMT{i}' for i in range(1, 7)]
avg_bill_positive = X[y == 1][all_bill_cols].to_numpy().flatten().mean()
avg_bill_all = X[all_bill_cols].to_numpy().flatten().mean()
print(f"Overall average BILL_AMT for positive class: {avg_bill_positive:.2f}")
print(f"Overall average BILL_AMT for entire dataset: {avg_bill_all:.2f}")

# PAY_AMT1 to PAY_AMT6
all_pay_cols = [f'PAY_AMT{i}' for i in range(1, 7)]
avg_pay_amt_positive = X[y == 1][all_pay_cols].to_numpy().flatten().mean()
avg_pay_amt_all = X[all_pay_cols].to_numpy().flatten().mean()
print(f"Overall average PAY_AMT for positive class: {avg_pay_amt_positive:.2f}")
print(f"Overall average PAY_AMT for entire dataset: {avg_pay_amt_all:.2f}")

# PAY_0, PAY_2 to PAY_6
all_pay_i_cols = [f'PAY_{i}' for i in [0, 2, 3, 4, 5, 6]]
avg_pay_i_positive = X[y == 1][all_pay_i_cols].to_numpy().flatten().mean()
avg_pay_i_all = X[all_pay_i_cols].to_numpy().flatten().mean()
print(f"Overall average PAY_I value for positive class: {avg_pay_i_positive:.2f}")
print(f"Overall average PAY_I value for entire dataset: {avg_pay_i_all:.2f}")
