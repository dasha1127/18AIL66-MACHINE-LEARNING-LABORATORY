import pandas as pd

# Load the dataset
df = pd.read_csv('oil-spill.csv', header=None)

# Identify and delete rows with duplicate data
print("Before removing duplicates:")
print(df.shape)

df.drop_duplicates(inplace=True)

print("After removing duplicates:")
print(df.shape)

# Identify and delete columns with a single value
print("Before removing columns with a single value:")
print(df.shape)

to_del = [col for col in df.columns if df[col].nunique() == 1]
df.drop(to_del, axis=1, inplace=True)

print("After removing columns with a single value:")
print(df.shape)
