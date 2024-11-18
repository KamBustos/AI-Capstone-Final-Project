import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the Dataset
file_path = r"C:\Users\kbust\Desktop\h3e2gb\Ai_data.csv"
data = pd.read_csv(file_path)

# Step 2: Inspect the Dataset
print("Dataset Shape:", data.shape)
print(data.head())
print(data.info())
print(data.describe())

# Step 3: Handle Missing Values
# Fill missing values with mean (or drop rows if preferred)
data.fillna(data.mean(numeric_only=True), inplace=True)

# Step 4: Feature Selection (Select relevant columns for prediction)
features = [
    "Age", "Overall Scote", "Potential Score", "Height", "Weight",
    "Acceleration", "Sprint Speed", "Reactions", "Vision", "Dribbling",
    "Finishing", "Short Passing", "Shot Power", "Stamina", "Aggression"
]
target = "Value"  # Predict the player's market value

X = data[features]
y = data[target]

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Build and Train a Random Forest Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 7: Evaluate the Model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# Step 8: Feature Importance
importance = model.feature_importances_
feature_importance = pd.DataFrame({'Feature': features, 'Importance': importance})
feature_importance.sort_values(by='Importance', ascending=False, inplace=True)

# Plot Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title("Feature Importance")
plt.show()

# Step 9: Save the Model for Future Use (Optional)
import joblib
joblib.dump(model, r"C:\Users\kbust\Desktop\h3e2gb\player_value_model.pkl")

print("Model saved successfully.")
