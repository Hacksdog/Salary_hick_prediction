import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv("Salary_Data.csv")
print("First 5 rows:")
print(df.head())

print("\nInfo:")
print(df.info())

print("\nSummary statistics:")
print(df.describe())

print("\nCorrelation between YearsExperience and Salary:")
print(df[["YearsExperience", "Salary"]].corr())

plt.scatter(df["YearsExperience"], df["Salary"])
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Experience vs Salary")
plt.show()

X = df[["YearsExperience"]]   
y = df["Salary"]              

model = LinearRegression()
model.fit(X, y)

m = model.coef_[0]
c = model.intercept_

print("\nModel equation:")
print(f"Salary = {m:.4f} * YearsExperience + {c:.4f}")

y_pred = model.predict(X)

rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

print(f"\nRMSE: {rmse:.4f}")
print(f"R-squared: {r2:.4f}")

new_exp = pd.DataFrame({"YearsExperience": [2, 5, 10]})

predicted_salary = model.predict(new_exp)

pred_df = new_exp.copy()
pred_df["Predicted_Salary"] = predicted_salary

print("\nPredictions for new experience values:")
print(pred_df)

pred_df.to_csv("salary_predictions.csv", index=False)
print("\nSaved predictions to salary_predictions.csv")
