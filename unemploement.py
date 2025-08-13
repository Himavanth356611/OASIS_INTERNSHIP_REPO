# Unemployment Rate Analysis in India

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
df1 = pd.read_csv("Unemployment in India.csv")
df2 = pd.read_csv("Unemployment_Rate_upto_11_2020.csv")

# View basic info
print("Dataset 1 Preview:")
print(df1.head())

print("\nDataset 2 Preview:")
print(df2.head())

# Clean column names
df2.columns = df2.columns.str.strip().str.replace(" ", "_")

# Convert date to datetime format
df2['Date'] = pd.to_datetime(df2['Date'], errors='coerce')

# Plot: Unemployment rate over time
plt.figure(figsize=(12, 6))
sns.lineplot(x='Date', y='Estimated_Unemployment_Rate%', data=df2, hue='Region', marker='o')
plt.title("Unemployment Rate Over Time by Region")
plt.xticks(rotation=45)
plt.ylabel("Unemployment Rate (%)")
plt.tight_layout()
plt.savefig("unemployment_trend_by_region.png")
plt.show()

# Heatmap of Unemployment Rate by Region and Month
df2['Month'] = df2['Date'].dt.strftime('%b-%Y')
pivot = df2.pivot("Region", "Month", "Estimated_Unemployment_Rate%")

plt.figure(figsize=(14, 8))
sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGnBu")
plt.title("Heatmap of Unemployment Rate by Region and Month")
plt.savefig("unemployment_heatmap.png")
plt.show()
