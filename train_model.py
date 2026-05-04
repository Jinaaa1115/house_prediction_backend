import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import pickle

# Load dataset
df = pd.read_csv("house_dataset.csv")

# Features and target
X = df[["Area_sqft", "Bedrooms"]]
y = df["Price"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"R² Score : {r2_score(y_test, y_pred)}")
print(f"MAE      : ${mean_absolute_error(y_test, y_pred)}")
print(f"Slope    : Area={model.coef_[0]}, Bedrooms={model.coef_[1]}")
print(f"Intercept: {model.intercept_}")

# Save model
with open("house_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\n✅ house_model.pkl saved successfully!")
