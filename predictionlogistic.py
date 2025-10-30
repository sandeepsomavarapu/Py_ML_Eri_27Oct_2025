#Step 1: Data Loading & Splitting
import pandas as pd
from sklearn.model_selection import train_test_split

# Define column names
columns = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
]

# Load the dataset
df = pd.read_csv('data.csv', header=None, names=columns, sep=r',\s*', engine='python', na_values='?')

# Separate Features (X) and Target (y)
X = df.drop('income', axis=1)
y = df['income']

# Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

from sklearn.impute import SimpleImputer

# Identify missing values
print('Missing values before imputation:')
print(X_train.isnull().sum())

# Impute Numerical Columns
num_imputer = SimpleImputer(strategy='mean')
X_train[['age']] = num_imputer.fit_transform(X_train[['age']])
X_test[['age']] = num_imputer.transform(X_test[['age']])

# Impute Categorical Columns
cat_imputer = SimpleImputer(strategy='most_frequent')
categorical_cols_with_missing = ['workclass', 'occupation', 'native-country']
X_train[categorical_cols_with_missing] = cat_imputer.fit_transform(X_train[categorical_cols_with_missing])
X_test[categorical_cols_with_missing] = cat_imputer.transform(X_test[categorical_cols_with_missing])

print('Missing values after imputation:')
print(X_train.isnull().sum())

# Drop 'education' as 'education-num' is its numerical representation
X_train = X_train.drop('education', axis=1)
X_test = X_test.drop('education', axis=1)

# One-Hot Encoding
categorical_cols = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)

# Align columns between training and testing sets
train_cols = X_train.columns
test_cols = X_test.columns

missing_in_test = set(train_cols) - set(test_cols)
for c in missing_in_test:
    X_test[c] = 0

missing_in_train = set(test_cols) - set(train_cols)
for c in missing_in_train:
    X_train[c] = 0
    
X_test = X_test[train_cols]

print("✅ Preprocessing complete. Train and test sets are now aligned.")
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

from sklearn.preprocessing import StandardScaler

# Define the scaler
scaler = StandardScaler()

# List of numerical columns to scale
numerical_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
print(f"Standardizing the following numerical columns: {numerical_cols}")

# Fit the scaler on training data and transform both train and test sets
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
print("✅ Training data standardized.")

X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
print("✅ Testing data standardized.")

# Optional: Show a sample of scaled values
print("\nSample of scaled training data:")
print(X_train[numerical_cols].head())

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Train Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f'Model Accuracy: {accuracy}')
# Load new data
new_df = pd.read_csv('my_data.csv', header=None, names=columns[:-1], sep=r',\s*', engine='python', na_values='?')

# Apply the same preprocessing
new_df[['age']] = num_imputer.transform(new_df[['age']])
new_df[categorical_cols_with_missing] = cat_imputer.transform(new_df[categorical_cols_with_missing])
new_df = new_df.drop('education', axis=1)
new_df = pd.get_dummies(new_df, columns=categorical_cols, drop_first=True)

# Align columns
train_cols = X_train.columns
new_cols = new_df.columns
missing_in_new = set(train_cols) - set(new_cols)
for c in missing_in_new:
    new_df[c] = 0
    
missing_in_train = set(new_cols) - set(train_cols)
for c in missing_in_train:
    # This case should ideally not happen if the new data has the same possible values
    # for categorical features. If it does, we drop the column.
    new_df = new_df.drop(c, axis=1)


new_df = new_df[train_cols]

new_df[numerical_cols] = scaler.transform(new_df[numerical_cols])

# --- Make Predictions ---
predictions = model.predict(new_df)
    
print("Predictions for the new data:")
print(predictions)