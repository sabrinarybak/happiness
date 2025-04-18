import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy.stats import ttest_ind

file_path = os.path.join("happiness_2019_cleaned.csv")
df = pd.read_csv(file_path)

# Räkna ut korrelationer
corr = df.corr(numeric_only=True)

# Visualisera med värmekarta
plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Korrelation mellan variabler")
plt.show()






# Dela in länder i två grupper: topp 10 och botten 10
top_10 = df.sort_values(by='Happiness', ascending=False).head(10)
bottom_10 = df.sort_values(by='Happiness').head(10)

print("Topp 10 lyckligaste länder:")
print(top_10[['Country', 'Happiness']])
print("\nBotten 10 minst lyckliga länder:")
print(bottom_10[['Country', 'Happiness']])

#Jämför deras indikationer
top_avg = top_10.mean(numeric_only=True)
bottom_avg = bottom_10.mean(numeric_only=True)

comparison = pd.DataFrame({'Top 10': top_avg, 'Bottom 10': bottom_avg})
comparison.plot(kind='bar', figsize=(12,6), title="Jämförelse av faktorer mellan lyckligaste och minst lyckliga länder")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()




#Uteslut Rank
faktorer = ['Happiness', 'GDP', 'SocialSupport', 'Health', 'Freedom', 'Generosity', 'Corruption']

top_avg = top_10[faktorer].mean()
bottom_avg = bottom_10[faktorer].mean()

# Skapa jämförelse-datablad
comparison = pd.DataFrame({'Top 10': top_avg, 'Bottom 10': bottom_avg})

# Plot
comparison.plot(kind='bar', figsize=(12,6), title="Jämförelse av faktorer mellan lyckligaste och minst lyckliga länder")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()




# Jämföra varje faktor mellan dessa två grupper.
for faktor in faktorer:
    stat, p = ttest_ind(top_10[faktor], bottom_10[faktor])
    signif = "✓" if p < 0.05 else "✗"
    print(f"{faktor}: p-värde = {p:.6f} ({'Signifikant' if p < 0.05 else 'Inte signifikant'}) {signif}")









