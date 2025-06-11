import pandas as pd
import os
from torch import nn, optim
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from torch import cuda
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.nn as nn   

# Path to the project directory
# Change this to your project directory
# os.chdir("/path/to/your/project/directory")
        
os.chdir("/Users/dmitrijkrysko/Desktop/revetg.com/GIT/toxic-comment-moderation")

df = pd.read_csv("data/train.csv")

tfidf = TfidfVectorizer(max_features=10000)
X = tfidf.fit_transform(df['comment_text']).toarray()
y = df[['toxic','severe_toxic','obscene','threat','insult','identity_hate']].values

class ToxicClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size//2)
        self.fc3 = nn.Linear(hidden_size//2, output_size)    
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()


    def data_preprocessing(self, X: np.ndarray, y: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=42)
        self.X_train = torch.tensor(X_train, dtype=torch.float32)
        self.X_test = torch.tensor(X_test, dtype=torch.float32)
        self.y_train = torch.tensor(y_train, dtype=torch.float32)
        self.y_test = torch.tensor(y_test, dtype=torch.float32)
        self.X_val = torch.tensor(X_val, dtype=torch.float32)
        self.y_val = torch.tensor(y_val, dtype=torch.float32)
        
        return self.X_train, self.X_test, self.y_train, self.y_test, self.X_val, self.y_val

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(input_data))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x) 
        x = self.fc3(x)
        return x
    
    def metrics(self, prediction: torch.Tensor, target: torch.Tensor) -> tuple[float, float, float, float]:
        pred_labels = (prediction.detach().cpu().numpy() > 0.5).astype(int)
        target_np = target.detach().cpu().numpy().astype(int)
        accuracy = accuracy_score(target_np, pred_labels)
        recall = recall_score(target_np, pred_labels, average="macro", zero_division=0)
        precision = precision_score(target_np, pred_labels, average="macro", zero_division=0)
        f1 = f1_score(target_np, pred_labels, average="macro", zero_division=0)
        return accuracy, recall, precision, f1
    
    def train_model(self, X_train: torch.Tensor, y_train: torch.Tensor, X_val: torch.Tensor, y_val: torch.Tensor, epochs: int = 10, learning_rate: float = 0.01):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)
        X_train = X_train.to(device)
        y_train = y_train.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        pos_counts = y_train.sum(axis=0)
        neg_counts = y_train.shape[0] - pos_counts
        pos_weight = torch.tensor(neg_counts / pos_counts, dtype=torch.float32)

        loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            prediction = self.forward(X_train)
            loss = loss_function(prediction, y_train)
            loss.backward()
            optimizer.step()
            train_metrics = self.metrics(prediction, y_train)
            self.eval()
            with torch.no_grad():
                val_prediction = self.forward(X_val)
                val_loss = loss_function(val_prediction, y_val)
                val_metrics = self.metrics(val_prediction, y_val)
            print(f'Epoch {epoch+1}, Train Loss: {loss.item()}, Validation Loss: {val_loss.item()},'
            f'Train Metrics: {train_metrics}, Validation Metrics: {val_metrics}')
        


    def evaluate(self, X_test: torch.Tensor, y_test: torch.Tensor) -> tuple[float, float, float, float]:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)
        X_test = X_test.to(device)
        y_test = y_test.to(device)
        prediction = self.forward(X_test)
        loss = nn.BCEWithLogitsLoss()(prediction, y_test)
        accuracy, recall, precision, f1 = self.metrics(prediction, y_test)
        
        print(f"Loss: {loss.item()}, Accuracy: {accuracy}, Recall: {recall}, Precision: {precision}, F1: {f1}")

if __name__ == "__main__":
    our_model = ToxicClassifier(input_size=10000, hidden_size=500, output_size=6)
    our_model.data_preprocessing(X, y)
    our_model.train_model(our_model.X_train, our_model.y_train, our_model.X_val, our_model.y_val, epochs=30, learning_rate=0.01)
    