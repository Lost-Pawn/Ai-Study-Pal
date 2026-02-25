import pandas as pd
import matplotlib.pyplot as plt
import os

data = pd.read_csv("data/raw_data.csv")
print(f"Successfully loaded {len(data)} rows")

text_cols = ["subject", "text", "difficulty"]
for col in text_cols:
    if col in data.columns:
        data[col] = data[col].astype(str).str.lower().str.strip()

clean_data = data.drop_duplicates(subset=["subject", "text", "difficulty"])

clean_data = clean_data.dropna(subset=["subject", "text", "study_hours_needed"])

os.makedirs("data", exist_ok=True)
clean_data.to_csv("data/clean_data.csv", index=False)  
print("Clean data saved!")


clean_data["subject"].value_counts().head(20).plot(
    kind="bar",
    figsize=(14,6),
    color="steelblue",
    edgecolor="black",
    title="Top 20 Topics by Entry Count"
)
plt.xlabel("Subject")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("data/clean_chart.png")
plt.show()
print("Chart saved!")

