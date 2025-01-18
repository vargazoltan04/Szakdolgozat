from model import CNN
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import json


transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),         # Horizontal flipping
    transforms.RandomRotation(30),             # Rotate by a random angle (30 degrees max)
    transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),  # Randomly resize and crop
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Random color adjustments
    transforms.ToTensor(),             
    transforms.Grayscale(num_output_channels=1),       # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))       # Normalize (adjust based on your data)
])

# Load the MNIST dataset
train_dataset = datasets.ImageFolder(root='../train_data/class', transform=transform)

indices = list(range(len(train_dataset)))
test_indices = indices[4::5]  # Every 5th element
train_indices = [i for i in indices if i not in test_indices]


# Get the class-to-index mapping
class_to_idx = train_dataset.class_to_idx  # Assuming `train_dataset` is available

# Reverse the mapping: index to class label
idx_to_class = {v: k for k, v in class_to_idx.items()}

# Save the predictions to a JSON file
with open('index_class_mapping.json', 'w') as json_file:
    json.dump(idx_to_class, json_file, indent=4)

# Step 3: Create Subsets
#train_dataset = Subset(train_dataset, train_indices)
#test_dataset = Subset(train_dataset, test_indices)

test_dataset = [train_dataset[i] for i in range(len(train_dataset)) if i in test_indices]
train_dataset = [train_dataset[i] for i in range(len(train_dataset)) if i not in test_indices]
#print(train_dataset.classes)
#Create data loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False)
    
# Initialize model
model = CNN()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.12, momentum=0.15)


# Training loop
#5 epoch, 64 batch => 98.21%
#8 epoch, 64 batch => 99.11%
#8 epoch, 32 batch => 99.55%
#9 epoch, 32 batch => 99.58%
#10 epoch, 32 batch => 99.28%
#10 epoch, 64 batch => 99.47%
#15 epoch, 32 batch, learning_rate_optimization => 99.85%
#15 epoch, 16 batch, lr = 0.1, 4 conv layer => 99.87%
#20 epoch, 32 batch, lr = 0.12, 3 conv layer => 99.89%
epochs = 10
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
    if abs(loss_previous - loss_current) < 0.1 and not learning_rate_lowered:
        #print(loss_previous)
        #print(loss_current)
        #print('lower')
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