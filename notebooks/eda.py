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

df['sum_labels'] = df[['toxic','severe_toxic','obscene','threat','insult','identity_hate']].sum(axis=1)
print("Полностью чистых комментариев:", (df['sum_labels']==0).sum())
print("Доля чистых:", ((df['sum_labels']==0).mean()*100).round(2), "%")

print(df.duplicated(subset=['comment_text']).sum())