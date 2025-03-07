from model import VGG16
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import json
import cv2
import numpy as np

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),            # Convert to tensor
        transforms.ToTensor(),
    ])

    time = datetime.now().strftime("%H:%M:%S")
    print(f"{time}: Loading ... ")

    train_dataset = datasets.ImageFolder(root="C:/Szakdoga/data", transform=transform)

    time = datetime.now().strftime("%H:%M:%S")
    print(f"{time}: Loading finished ... ")




    indices = list(range(len(train_dataset)))
    test_indices = set(indices[4::5])
    train_indices = set([i for i in indices if i not in test_indices])

    test_indices = list(test_indices)
    train_indices = list(train_indices)

    # Get the class-to-index mapping
    class_to_idx = train_dataset.class_to_idx  # Assuming `train_dataset` is available

    # Reverse the mapping: index to class label
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # Save the predictions to a JSON file
    #with open('index_class_mapping.json', 'w') as json_file:
    #    json.dump(idx_to_class, json_file, indent=4)

    time = datetime.now().strftime("%H:%M:%S")
    print(f"{time}: Separating train/test ... ")

    # Create subsets directly
    test_dataset = Subset(train_dataset, test_indices)
    train_dataset = Subset(train_dataset, train_indices)

    time = datetime.now().strftime("%H:%M:%S")
    print(f"{time}: Separating train/test finished... ")


    time = datetime.now().strftime("%H:%M:%S")
    print(f"{time}: Creating data loaders ... ")

    train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True, num_workers=12, pin_memory=True, prefetch_factor=128)
    test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False, num_workers=12, pin_memory=True, prefetch_factor=128)

    #time = datetime.now().strftime("%H:%M:%S")
    #print(f"{time}: Creating data loaders finished... ")
    
    # Initialize model
    model = VGG16(58)
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.005, weight_decay=0.005, momentum=0.08)


    time = datetime.now().strftime("%H:%M:%S")
    print(f"{time}: Start training ... ")
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
        time = datetime.now().strftime("%H:%M:%S")
        print(f"{time}: Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

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
    time = datetime.now().strftime("%H:%M:%S")
    print(f"{time}: Accuracy: {100 * correct / total}%")

if __name__ == '__main__':
    main()