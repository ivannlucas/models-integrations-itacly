import torch.nn as nn

# Definición de la Arquitectura MEJORADA
class PasteurizationANN(nn.Module):
    def __init__(self, input_size):
        super(PasteurizationANN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        return self.network(x)
