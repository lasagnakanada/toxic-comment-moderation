import pandas as pd

df = pd.read_csv("data/train.csv")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

tfidf = TfidfVectorizer(max_features=10000)
X = tfidf.fit_transform(df['comment_text'])

y = df[['toxic','severe_toxic','obscene','threat','insult','identity_hate']]
model = OneVsRestClassifier(LogisticRegression())
model.fit(X, y)


y_pred = model.predict(X)

