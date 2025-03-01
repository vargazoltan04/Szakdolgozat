from model import VGG16
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import json
import cv2
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  
    transforms.Resize((64, 64)),            # Convert to tensor
    transforms.ToTensor(),
])
print("Loading ... ")
# Load the MNIST dataset
train_dataset = datasets.ImageFolder(root='../../train_data/data/', transform=transform)
indices = list(range(len(train_dataset)))
test_indices = indices[4::5]  # Every 5th element
train_indices = [i for i in indices if i not in test_indices]


# Get the class-to-index mapping
class_to_idx = train_dataset.class_to_idx  # Assuming `train_dataset` is available

# Reverse the mapping: index to class label
idx_to_class = {v: k for k, v in class_to_idx.items()}

# Save the predictions to a JSON file
#with open('index_class_mapping.json', 'w') as json_file:
#    json.dump(idx_to_class, json_file, indent=4)

# Step 3: Create Subsets
#train_dataset = Subset(train_dataset, train_indices)
#test_dataset = Subset(train_dataset, test_indices)

test_dataset = [train_dataset[i] for i in range(len(train_dataset)) if i in test_indices]
train_dataset = [train_dataset[i] for i in range(len(train_dataset)) if i in train_indices]
#print(train_dataset.classes)
#Create data loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)
    
# Initialize model
model = VGG16(58)
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.005, weight_decay=0.005, momentum=0.08)



print("Start training ... ")
epochs = 4
loss_previous = 999999
learning_rate_lowered = False
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        
        # Zero the gradients
        optimizer.zero_grad()
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        
    loss_current = running_loss/len(train_loader)
    if abs(loss_previous - loss_current) < 0.05 and not learning_rate_lowered:
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
    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

# Save the model weights (state dictionary)
torch.save(model.state_dict(), 'model_weights.pth')
print(f"Accuracy: {100 * correct / total}%")