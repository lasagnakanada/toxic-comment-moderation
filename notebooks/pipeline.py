import pandas as pd
import os
from torch import nn, optim
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
        
os.chdir("/Users/dmitrijkrysko/Desktop/revetg.com/GIT/toxic-comment-moderation")

df = pd.read_csv("data/train.csv")

tfidf = TfidfVectorizer(max_features=10000)
X = tfidf.fit_transform(df['comment_text']).toarray()
y = df[['toxic','severe_toxic','obscene','threat','insult','identity_hate']].values

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=42)


model = nn.Sequential(
    nn.Linear(10000, 1000),
    nn.Sigmoid(),
    nn.Linear(1000, 1000),
    nn.ReLU(),
    nn.Linear(1000, 6),
    nn.Sigmoid()
)

loss_function = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)






X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

def train_model(model, loss_function, optimizer, X_train, y_train, X_test, y_test, n_epochs):
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = loss_function(y_pred, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            y_pred = model(X_test)
            test_loss = loss_function(y_pred, y_test)
            print(f"Epoch {epoch+1} - Loss: {loss.item()}, Test Loss: {test_loss.item()}")

train_model(model, loss_function, optimizer, X_train, y_train, X_test, y_test, n_epochs=10)










