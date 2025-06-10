import pandas as pd

df = pd.read_csv("data/train.csv")

print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())
for col in df.columns[2:]:
    print(f'{col}: {df[col].sum()}')

print(df[df['toxic'] == 1]['comment_text'].sample(3).values)
print(df[df['toxic'] == 0]['comment_text'].sample(3).values)


df['text_len'] = df['comment_text'].apply(len)
df['text_len'].hist(bins=50)