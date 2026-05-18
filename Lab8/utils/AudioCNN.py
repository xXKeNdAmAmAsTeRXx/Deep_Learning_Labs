from torch import nn
import torch




import torch.nn as nn

class AudioCNN(nn.Module):
    """An Audio Data classifier for 1 second 8 kHz audio Input"""
    def __init__(self, dropout_rate: float = 0.3, num_classes: int = 10):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels=1,  out_channels=8,  kernel_size=13, padding='valid'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),
            nn.Dropout(dropout_rate),

            # Block 2
            nn.Conv1d(in_channels=8,  out_channels=16, kernel_size=11,  padding='valid'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),
            nn.Dropout(dropout_rate),

            # Block 3
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=9,  padding='valid'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),
            nn.Dropout(dropout_rate),

            # Block 4
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7,  padding='valid'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),
            nn.Dropout(dropout_rate),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(6080, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)

        return x