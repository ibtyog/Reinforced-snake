import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class LinearNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(17, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 4)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './saved_models'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)