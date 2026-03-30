# Student Performance Prediction Project
import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

2. Create Sample Dataset 
data = {
    "study_time": [1, 2, 3, 4, 5, 2, 3, 4, 1, 5],
    "attendance": [60, 70, 80, 90, 95, 75, 85, 92, 65, 98],
    "previous_score": [50, 55, 65, 70, 80, 60, 68, 75, 52, 85],
    "parent_education": ["high", "medium", "high", "low", "medium", "high", "low", "medium", "high", "low"],
    "internet": ["yes", "yes", "no", "yes", "yes", "no", "yes", "yes", "no", "yes"],
    "final_score": [55, 60, 70, 80, 90, 65, 75, 85, 58, 95]
}
df = pd.DataFrame(data)
print("\nDataset Preview:")
print(df.head())

3. Encode Categorical Data
le = LabelEncoder()
df["parent_education"] = le.fit_transform(df["parent_education"])
df["internet"] = le.fit_transform(df["internet"])
4. Split Features and Target
X = df.drop("final_score", axis=1)
y = df["final_score"]
5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

6. Train Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
7. Predictions
y_pred = model.predict(X_test)
8. Evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\nModel Performance:")
print("Mean Absolute Error:", mae)
print("R2 Score:", r2)
 9. Test with Custom Input
sample = np.array([[3, 85, 70, 1, 1]])  # example input
prediction = model.predict(sample)
print("\nPredicted Student Score:", prediction[0])
10. Visualization
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Scores")
plt.ylabel("Predicted Scores")
plt.title("Actual vs Predicted Scores")
plt.show()