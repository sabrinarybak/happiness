import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np

# Load cleaned dataset
df = pd.read_csv("happiness_2019_cleaned.csv")

# Top and bottom 10 happiest countries
top_10 = df.sort_values(by='Happiness', ascending=False).head(10)
bottom_10 = df.sort_values(by='Happiness').head(10)
data = pd.concat([top_10, bottom_10]).reset_index(drop=True)

# Features to test
features = ['GDP', 'SocialSupport', 'Health', 'Freedom', 'Generosity', 'Corruption']
y = data['Happiness']

# Plot regression for each feature
for feature in features:
    X = data[[feature]]
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)

    # Plot
    plt.figure(figsize=(6, 4))
    plt.scatter(X, y, color='blue', label='Data')
    plt.plot(X, y_pred, color='red', label='Regression line')
    plt.title(f"Happiness vs {feature} (R² = {r2:.2f})")
    plt.xlabel(feature)
    plt.ylabel("Happiness")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()