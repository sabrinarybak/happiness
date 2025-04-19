import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("happiness_2019_cleaned.csv")
top_10=df.sort_values(by='Happiness', ascending=False).head(10)
bottom_10=df.sort_values(by='Happiness').head(10)
data = pd.concat([top_10,bottom_10])
features= ['GDP', 'SocialSupport', 'Health', 'Freedom', 'Generosity', 'Corruption']
X = data[features]
y = data['Happiness']
model = LinearRegression()
model.fit(X,y)
y_pred = model.predict(X)

# Print regression coefficients
print("Regression coefficients (impact of each factor):")
for name, coef in zip(features, model.coef_):
    print(f"{name}: {coef:.3f}")

# Print intercept
print(f"\nIntercept: {model.intercept_:.3f}")

# Model evaluation
print("\nModel performance:")
print(f"R² score: {r2_score(y, y_pred):.3f}")
print(f"Mean Squared Error: {mean_squared_error(y, y_pred):.3f}")

# Visualize factor importance
plt.figure(figsize=(8, 6))
plt.barh(features, model.coef_)
plt.axvline(0, color='gray', linestyle='--')
plt.title("Importance of Each Factor (Regression Coefficients)")
plt.xlabel("Coefficient")
plt.ylabel("Factor")
plt.tight_layout()
plt.show()



