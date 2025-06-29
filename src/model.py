# INITIALIZING TRAINING

import pandas as pd
import os
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.nn as nn
from tqdm import tqdm, trange
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import shutil
import json
from src.config import EMBEDDINGS_PATH, LABELS_PATH
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#shutil.copy('/content/drive/MyDrive/embeddings.pt', '/content/embeddings.pt')
# Теперь загружаем уже с локального

X = torch.load(EMBEDDINGS_PATH)
y = np.load(LABELS_PATH)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, pos_weight=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, pos_weight=self.pos_weight, reduction='none'
        )
        # Get probabilities for focal term
        probs = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        loss = focal_weight * bce_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class ToxicClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int,):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)

        self.dropout = nn.Dropout(0.4)
        self.relu = nn.ReLU()

        self.class_thresholds = [0.5] * output_size

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:

        out1 = self.relu(self.norm1(self.fc1(input_data)))
        out2 = self.dropout(out1)

        out3 = self.relu(self.norm2(self.fc2(out2)+out2))
        out4 = self.dropout(out3)

        out5 = self.relu(self.norm3(self.fc3(out4)+out4))
        out6 = self.fc4(out5)
        return out6

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


    
    def metrics(self, prediction: torch.Tensor, target: torch.Tensor, thresholds=None) -> tuple[float, float, float, float]:
        probs = torch.sigmoid(prediction).detach().cpu().numpy()
        target_np = target.detach().cpu().numpy().astype(int)
        if thresholds is None:
            thresholds = [0.7] * probs.shape[1]
        pred_labels = np.array([
            [p > t for p, t in zip(sample, thresholds)]
            for sample in probs
        ]).astype(int)
        accuracy = accuracy_score(target_np, pred_labels)
        recall = recall_score(target_np, pred_labels, average="macro", zero_division=0)
        precision = precision_score(target_np, pred_labels, average="macro", zero_division=0)
        f1 = f1_score(target_np, pred_labels, average="macro", zero_division=0)
        return accuracy, recall, precision, f1

    def train_model(self, train_loader, val_loader, epochs: int = 10, learning_rate: float = 0.001):
      self.to(device)
      optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=1e-6)
      scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
      pos_counts = self.y_train.sum(axis=0)
      neg_counts = self.y_train.shape[0] - pos_counts
      max_weight = 10
      pos_weight = torch.tensor(np.clip(neg_counts / pos_counts, 1, max_weight), dtype=torch.float32).to(device)

      loss_function = FocalLoss(gamma=2, pos_weight=pos_weight)

      best_val_loss = float('inf')
      best_val_f1 = 0.0
      patience = 20
      counter = 0

      for epoch in range(epochs):
          # --- TRAINING ---
          self.train()
          train_losses = []
          train_logits, train_targets = [], []


          for X_batch, y_batch in train_loader:
              X_batch = X_batch.to(device)
              y_batch = y_batch.to(device)

              optimizer.zero_grad()
              logits = self.forward(X_batch)
              loss = loss_function(logits, y_batch)
              loss.backward()
              optimizer.step()
              train_losses.append(loss.item())
              train_logits.append(logits.detach().cpu())
              train_targets.append(y_batch.detach().cpu())

          train_logits = torch.cat(train_logits)
          train_loss_mean = np.mean(train_losses)
          train_targets = torch.cat(train_targets)
          train_acc, train_rec, train_prec, train_f1 = self.metrics(train_logits, train_targets, thresholds=[0.5,0.5,0.5,0.5,0.5,0.5])
        # --- VALIDATION ---
          self.eval()
          val_logits, val_targets = [], []

          val_losses = []

          with torch.no_grad():
              for X_val_batch, y_val_batch in val_loader:
                  X_val_batch = X_val_batch.to(device)
                  y_val_batch = y_val_batch.to(device)
                  logits = self.forward(X_val_batch)
                  val_logits.append(logits.detach().cpu())
                  val_targets.append(y_val_batch.detach().cpu())
                  batch_loss = loss_function(logits, y_val_batch)
                  val_losses.append(batch_loss.item())
          val_logits = torch.cat(val_logits)
          val_targets= torch.cat(val_targets)
          val_loss = np.mean(val_losses)
          val_acc, val_rec, val_prec, val_f1 = self.metrics(val_logits, val_targets, thresholds=self.class_thresholds)

          print(
              f"\nEpoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss_mean:.4f} | Val Loss: {val_loss:.4f}\n"
              f"Train: Acc: {train_acc:.4f} | Rec: {train_rec:.4f} | Prec: {train_prec:.4f} | F1: {train_f1:.4f}\n"
              f"Val:   Acc: {val_acc:.4f} | Rec: {val_rec:.4f} | Prec: {val_prec:.4f} | F1: {val_f1:.4f}\n"
              f"Val class sums (GT): {val_targets.sum(dim=0).numpy()}"
          )

          # --- EARLY STOPPING ---
          if val_f1 > best_val_f1:
              best_val_f1 = val_f1
              counter = 0
              # можно здесь torch.save(self.state_dict(), 'best_model.pth')
          else:
              counter += 1

          if counter >= patience:
              print(f"Early stopping at epoch {epoch+1}")
              break

          scheduler.step()

      print("Train targets sum per class:", self.y_train.sum(dim=0))
      print("Val targets sum per class:", self.y_val.sum(dim=0))



    def evaluate(self, X_test: torch.Tensor, y_test: torch.Tensor) -> tuple[float, float, float, float]:
        self.to(device)
        X_test = X_test.to(device)
        y_test = y_test.to(device)
        prediction = self.forward(X_test)
        loss = FocalLoss(gamma=2, pos_weight=None)(prediction, y_test)

        accuracy, recall, precision, f1 = self.metrics(prediction, y_test)

        print(f"Loss: {loss.item()}, Accuracy: {accuracy}, Recall: {recall}, Precision: {precision}, F1: {f1}")

    def find_best_thresholds(model, X_val, y_val):
        model.eval()
        with torch.no_grad():
            logits = model(X_val.to(device)).cpu()
            probs = torch.sigmoid(logits).numpy()
        best_thresholds = []
        for i in range(probs.shape[1]):
            best_f1 = 0
            best_thr = 0.5
            for thr in np.arange(0.1, 0.91, 0.01):
                preds = (probs[:, i] > thr).astype(int)
                f1 = f1_score(y_val[:, i], preds, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thr = thr
            best_thresholds.append(best_thr)
        return best_thresholds
          