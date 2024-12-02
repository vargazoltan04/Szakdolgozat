import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F

transform = transforms.Compose([
    transforms.Resize((64, 64)),  
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# Load the MNIST dataset
train_dataset = datasets.ImageFolder(root='../../train_data/class', transform=transform)

indices = list(range(len(train_dataset)))
test_indices = indices[4::5]  # Every 5th element
train_indices = [i for i in indices if i not in test_indices]

# Step 3: Create Subsets
#train_dataset = Subset(train_dataset, train_indices)
#test_dataset = Subset(train_dataset, test_indices)

test_dataset = [train_dataset[i] for i in range(len(train_dataset)) if i in test_indices]
train_dataset = [train_dataset[i] for i in range(len(train_dataset)) if i not in test_indices]
#print(train_dataset.classes)
#Create data loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(in_features=32 * 8 * 8, out_features=128)  # Adjust for input size
        self.fc2 = nn.Linear(in_features=128, out_features=94)

        self.softmax = nn.LogSoftmax(dim = 1)

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
        
        # Flatten the tensor
        x = x.view(x.size(0), -1)  # Flatten for the fully connected layer
        
        # Fully connected layers with activation
        x = F.relu(self.fc1(x))
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
    
# Initialize model
model = CNN()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training loop
#5 epoch, 64 batch => 98.21%
#8 epoch, 64 batch => 99.11%
#8 epoch, 32 batch => 99.55%
#9 epoch, 32 batch => 99.58%
#10 epoch, 32 batch => 99.28%
#10 epoch, 64 batch => 99.47%
#15 epoch, 32 batch, learning_rate_optimization => 99.85%
epochs = 15
loss_previous = 999999
learning_rate_lowered = False
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Calculate loss
        loss = criterion(outputs, labels)

        
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        
    loss_current = running_loss/len(train_loader)
    if abs(loss_previous - loss_current) < 0.03 and not learning_rate_lowered:
        print(loss_previous)
        print(loss_current)
        print('lower')
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr']/10
        
        learning_rate_lowered = True

    loss_previous = loss_current
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

# Evaluate the model
model.eval()
correct = 0
total = 0

for images, labels in test_loader:
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

# Save the model weights (state dictionary)
torch.save(model.state_dict(), 'model_weights.pth')
print(f"Accuracy: {100 * correct / total}%")