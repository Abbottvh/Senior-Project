import Senior_Project_Data_Organization

import torch
import torch.nn as nn
import torch.nn.functional as F
#defining the CNN

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

model = ConvNet().to(Senior_Project_Data_Organization.device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=Senior_Project_Data_Organization.learning_rate)

n_total_steps = len(Senior_Project_Data_Organization.train_loader)

for epoch in range(Senior_Project_Data_Organization.num_epochs):

    running_loss = 0.0

    for i, (images, labels) in enumerate(Senior_Project_Data_Organization.train_loader):
        images = images.to(Senior_Project_Data_Organization.device)
        labels = labels.to(Senior_Project_Data_Organization.device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()

print(f'[{epoch + 1}] loss: {running_loss /n_total_steps:.3f}')

print('Finished Training')
PATH = './cnn.pth'
torch.save(model.state_dict(), PATH)