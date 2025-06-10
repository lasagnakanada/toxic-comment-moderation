import pandas as pd
import os
from torch import nn, optim
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from torch import cuda
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
# Path to the project directory
# Change this to your project directory
# os.chdir("/path/to/your/project/directory")
        
os.chdir("/Users/dmitrijkrysko/Desktop/revetg.com/GIT/toxic-comment-moderation")

df = pd.read_csv("data/train.csv")

tfidf = TfidfVectorizer(max_features=10000)
X = tfidf.fit_transform(df['comment_text']).toarray()
y = df[['toxic','severe_toxic','obscene','threat','insult','identity_hate']].values

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=42)


model = nn.Sequential(
    nn.Linear(10000, 5000),
    nn.Sigmoid(),
    nn.Dropout(p=0.4),
    nn.Linear(5000, 2000),
    nn.Sigmoid(),
    nn.Dropout(p=0.4),
    nn.Linear(2000, 6),
)

batch_size = 128
loss_function = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
if cuda.is_available():
    model = model.cuda()
    loss_function = loss_function.cuda()




X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(TensorDataset(X_test,  y_test),  batch_size=batch_size, shuffle=False)


def train_model(model, loss_function, optimizer, train_loader, test_loader, n_epochs):
    for epoch in range(n_epochs):
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            if cuda.is_available():
                X_batch = X_batch.cuda()
                y_batch = y_batch.cuda()

            # Forward pass
            model.train()
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = loss_function(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)


        model.eval()
        test_loss = 0.0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                if cuda.is_available():
                    X_batch = X_batch.cuda()
                    y_batch = y_batch.cuda()
                y_pred = model(X_batch)
                loss = loss_function(y_pred, y_batch)
                test_loss += loss.item() * X_batch.size(0)
                all_preds.append(torch.sigmoid(y_pred).cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())
            test_loss /= len(test_loader.dataset)
        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)
        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        print(f'Epoch {epoch+1} - Accuracy: {accuracy_score(all_targets, all_preds > 0.5)}')
        print(f'Epoch {epoch+1} - F1 Score: {f1_score(all_targets, all_preds > 0.5, average="weighted")}')
        print(f'Epoch {epoch+1} - Precision: {precision_score(all_targets, all_preds > 0.5, average="weighted")}')
        print(f'Epoch {epoch+1} - Recall: {recall_score(all_targets, all_preds > 0.5, average="weighted")}')


train_model(model, loss_function, optimizer, test_loader, train_loader, n_epochs=5)

