import os 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 

def get_len(text):
    text = str(text)
    text = text.split()
    return len(text)

filePath = "data/hindi_wikipedia_articles_172k.csv"
diskSizeMB = os.path.getsize(filePath)/(1024*1024)
print(f"On disk size: {diskSizeMB:.2f} MB.")

df = pd.read_csv(filePath)
memorySizeMB = df.memory_usage(deep=True).sum()/(1024*1024)
print(f"In memory size: {memorySizeMB:.2f} MB.")
print(f"Number of texts present in dataset: {df.shape[0]}.")
df["textLength"] = df["text"].apply(get_len)

print(f"Column names: {df.columns}")
print(df.head())

textAverageLength = df["textLength"].mean()
textMedianLength = df["textLength"].median()
textModeLength = df["textLength"].mode()

print(f"Average text length: {int(textAverageLength)}.")
print(f"Median text length: {int(textMedianLength)}.")
print(f"Mode text length: {int(textModeLength[0])}.")

sns.histplot(df["textLength"], bins=100, kde=True)
plt.xscale("log")
plt.xlabel("Text length (words)")
plt.ylabel("Count (text samples)")
plt.title("Distribution of text lengths (log scale)")
plt.show()