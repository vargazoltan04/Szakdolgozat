import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        #self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(in_features=32 * 8 * 8, out_features=128)  # Adjust for input size
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=128, out_features=26)

        #self.fc1 = nn.Linear(3 * 64 * 64, 2048)  # Input layer
        #self.fc2 = nn.Linear(2048, 1024)      # Hidden layer
        #self.fc3 = nn.Linear(1024, 512)       # Output layer
        #self.fc4 = nn.Linear(512, 256)  # Input layer
        #self.fc5 = nn.Linear(256, 128)  # Input layer
        #self.fc6 = nn.Linear(128, 94)  # Input layer
        #self.relu = nn.ReLU()              # Activation function
        #self.softmax = nn.LogSoftmax(dim=1)  # Softmax for output probabilities

    def forward(self, x):

        #Convolutional layers with ReLU activation and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        #x = self.pool(F.relu(self.conv4(x)))
        
        # Flatten the tensor
        x = x.view(x.size(0), -1)  # Flatten for the fully connected layer
        # Fully connected layers with activation
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)  # No activation for output (handled in loss)
        return x
    
        #x = x.view(-1, 3 * 64 * 64)  # Flatten the input
        #x = self.relu(self.fc1(x))
        #x = self.relu(self.fc2(x))
        #x = self.relu(self.fc3(x))
        #x = self.relu(self.fc4(x))
        #x = self.relu(self.fc5(x))
        #x = self.softmax(self.fc6(x))
        #return x