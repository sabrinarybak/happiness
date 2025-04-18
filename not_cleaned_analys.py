import pandas as pd
import os
import kagglehub
import seaborn as sns
import matplotlib.pyplot as plt

# Ange sökvägen till mappen där filerna laddats ner
path = kagglehub.dataset_download("unsdsn/world-happiness")

# Skapa full sökväg till just 2019-filen
file_2019 = os.path.join(path, "2019.csv")

# Läs in 2019-års data
df = pd.read_csv(file_2019)

# Utforska rådatan
print(df.head())
print(df.info())
print(df.describe())