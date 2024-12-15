import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionModel(nn.Module):
    def __init__(self, state_size, num_actions):
        super(AttentionModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.fc_q = nn.Linear(64, num_actions)

    def forward(self, state_tensor):
        encoded_output = self.encoder(state_tensor)
        q_values = self.fc_q(encoded_output)
        return q_values
