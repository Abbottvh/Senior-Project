import torch
import torch.nn as nn
import torch.nn.functional as F
from Senior_Project_Data_Organization import device, test_loader

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(64*4*4, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        # N, 3, 32, 32
        x = F.relu(self.conv1(x)) # -> N, 32, 30, 30
        x = self.pool(x) # -> N, 32, 15, 15
        x = F.relu(self.conv2(x)) # -> N, 64, 13, 13
        x = self.pool(x) # -> N, 64, 6, 6
        x = F.relu(self.conv3(x)) # -> N, 64, 4, 4
        x = torch.flatten(x, 1) # -> N, 1024
        x = F.relu(self.fc1(x)) # -> N, 64

        x = self.fc2(x) # -> N, 10
        return x
    
#Loading and testing the CNN on the test data
PATH = './cnn.pth' 
loaded_model = ConvNet()

loaded_model.load_state_dict(torch.load(PATH)) # it takes the loaded dictionary, not the path file itself
loaded_model.to(device)
loaded_model.eval()

with torch.no_grad():
    n_correct = 0
    n_correct2 = 0
    n_samples = len(test_loader.dataset)

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = loaded_model(images)

        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_correct += (predicted == labels).sum().item()

        outputs2 = loaded_model(images)
        _, predicted2 = torch.max(outputs2, 1)
        n_correct2 += (predicted2 == labels).sum().item()

acc = 100.0 * n_correct / n_samples
print(f'Accuracy of the model: {acc} %')

acc = 100.0 * n_correct2 / n_samples
print(f'Accuracy of the loaded model: {acc} %')