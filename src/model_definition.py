import torch
import torch.nn as nn

class ToxicClassifier(nn.Module):
    """
    Defines the neural network architecture for inference.
    This class is separate from the training script to ensure the production
    environment only contains code necessary for inference.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
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

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        out1 = self.relu(self.norm1(self.fc1(input_data)))
        out2 = self.dropout(out1)
        out3 = self.relu(self.norm2(self.fc2(out2) + out2))
        out4 = self.dropout(out3)
        out5 = self.relu(self.norm3(self.fc3(out4) + out4))
        out6 = self.fc4(out5)
        return out6