import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the Dataset
file_path = r"C:\Users\kbust\Desktop\AICAPPROJ\AI-Capstone-Final-Project\Ai_data.csv"
data = pd.read_csv(file_path)

# Step 2: Clean and Preprocess Data
# Convert 'Height' from string to numeric (extracting cm)
def parse_height(height):
    try:
        # Extract the cm part
        cm = height.split("cm")[0]
        return int(cm)
    except:
        return np.nan

data['Height'] = data['Height'].apply(parse_height)

# Convert 'Value', 'Wage', and 'Release Clause' to numeric (remove currency symbols and convert to millions)
def parse_money(value):
    try:
        value = value.replace('€', '').replace('M', '').replace('K', '')
        if 'M' in value:  # Convert millions
            return float(value) * 1e6
        elif 'K' in value:  # Convert thousands
            return float(value) * 1e3
        return float(value)
    except:
        return np.nan

data['Value'] = data['Value'].apply(parse_money)
data['Wage'] = data['Wage'].apply(parse_money)
data['Release Clause'] = data['Release Clause'].apply(parse_money)

# Drop rows with missing values in important features
data.dropna(subset=['Height', 'Value', 'Wage'], inplace=True)

# Step 3: Feature Selection
features = [
    "Age", "Overall Scote", "Potential Score", "Height", "Acceleration",
    "Sprint Speed", "Reactions", "Vision", "Dribbling", "Finishing",
    "Short Passing", "Shot Power", "Stamina", "Aggression"
]
target = "Value"  # Predict player's market value

# Ensure all features are numeric
X = data[features].apply(pd.to_numeric, errors='coerce')
y = data[target]

# Drop rows with missing values again to ensure consistency
print("Features (X) Shape Before Dropping Missing:", X.shape)
print("Target (y) Shape Before Dropping Missing:", y.shape)

X = X.dropna()
y = y[X.index]  # Align y with X

print("Features (X) Shape After Dropping Missing:", X.shape)
print("Target (y) Shape After Dropping Missing:", y.shape)

# Ensure there are enough samples
if X.shape[0] == 0:
    raise ValueError("No data samples remain after preprocessing. Check data cleaning steps.")

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Confirm split shapes
print("Training Features Shape:", X_train.shape)
print("Testing Features Shape:", X_test.shape)
print("Training Target Shape:", y_train.shape)
print("Testing Target Shape:", y_test.shape)

# Step 5: Build and Train a Random Forest Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Evaluate the Model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# Step 7: Feature Importance
importance = model.feature_importances_
feature_importance = pd.DataFrame({'Feature': features, 'Importance': importance})
feature_importance.sort_values(by='Importance', ascending=False, inplace=True)

# Plot Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title("Feature Importance")
plt.show()
