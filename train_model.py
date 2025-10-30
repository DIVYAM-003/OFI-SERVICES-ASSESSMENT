import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import glob

print("Starting model training process...")

# --- 1. Load Data ---
# Use glob to find all CSV files in the 'data' directory
data_files = glob.glob("data/*.csv")
if not data_files:
    print("Error: No CSV files found in the 'data' directory.")
    print("Please create a 'data' folder and place the 7 CSV files inside it.")
    exit()

print(f"Found {len(data_files)} CSV files. Loading...")

# Load all datasets into a dictionary
data = {}
for f in data_files:
    # Use a simple key, e.g., 'orders' from 'data/orders.csv'
    key = f.split('/')[-1].split('\\')[-1].replace('.csv', '')
    data[key] = pd.read_csv(f)
    print(f"Loaded {key}.csv")

# --- 2. Merge & Engineer Data ---
print("Merging datasets...")
try:
    # Merge the 3 key datasets for modeling
    df = data['orders'].merge(data['delivery_performance'], on='Order_ID')
    df = df.merge(data['routes_distance'], on='Order_ID')
except KeyError as e:
    print(f"Error: Missing required file. Could not find {e}.")
    print("Make sure 'orders.csv', 'delivery_performance.csv', and 'routes_distance.csv' are in the 'data' folder.")
    exit()

print("Data merged successfully.")

# --- 3. Feature Engineering ---
# Create the target variable 'is_delayed'
# 1 if 'Delayed', 0 if 'On-Time'
df['is_delayed'] = df['Delivery_Status'].apply(lambda x: 1 if x in ['Slightly-Delayed', 'Severely-Delayed'] else 0)

# Define feature and target
TARGET = 'is_delayed'
# Select the features for the model
FEATURES = [
    'Customer_Segment',
    'Priority',
    'Product_Category',
    'Origin',
    'Destination',
    'Special_Handling',
    'Promised_Delivery_Days',
    'Carrier',
    'Distance_KM',
    'Traffic_Delay_Minutes',
    'Weather_Impact'
]

X = df[FEATURES]
y = df[TARGET]

print(f"Features selected: {FEATURES}")

# --- 4. Preprocessing Pipeline ---
print("Building preprocessing pipeline...")

# Define numerical and categorical features
numerical_features = ['Promised_Delivery_Days', 'Distance_KM', 'Traffic_Delay_Minutes']
categorical_features = [
    'Customer_Segment',
    'Priority',
    'Product_Category',
    'Origin',
    'Destination',
    'Special_Handling',
    'Carrier',
    'Weather_Impact'
]

# Create transformers
# For numerical data: impute missing values with the median, then scale
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# For categorical data: impute missing values with a 'missing' string, then one-hot encode
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine transformers using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# --- 5. Model Training ---
print("Splitting data and training model...")
# Create the full model pipeline: preprocess, then classify
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10))
])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train the model
model_pipeline.fit(X_train, y_train)

print("Model training complete.")

# --- 6. Model Evaluation ---
print("Evaluating model...")
y_pred = model_pipeline.predict(X_test)

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=['On-Time', 'Delayed']))

print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))
print("(Rows: Actual, Columns: Predicted)")

# --- 7. Save Model ---
model_filename = 'delivery_model.pkl'
joblib.dump(model_pipeline, model_filename)

print(f"\nModel successfully trained and saved as '{model_filename}'!")
print("You can now run 'streamlit run app.py' to launch the application.")
