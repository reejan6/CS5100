import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

# load in audio data
filename = 'Audio/features+labels.npy'

with open(filename, 'rb') as f:
    X_train = np.load(f)
    X_val = np.load(f)
    X_test = np.load(f)
    y_train = np.load(f).astype(np.int64)
    y_val = np.load(f).astype(np.int64)
    y_test = np.load(f).astype(np.int64)

# create Tensor datasets
train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
val_data = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

batch_size = 50

# shuffling and batching data
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
valid_loader = DataLoader(val_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

# define parameters
num_classes = 8
dropout = 0.5
lr = 0.0001

# Audio CNN 
class Audio_CNN(nn.Module):
    def __init__(self, num_classes, dropout):
        super().__init__() 
        
        # Define the CNN layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        # Define BatchNorm layers
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        # Define MaxPooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Define the fully connected layers
        self.fc1 = None  
        self.fc2 = nn.Linear(128, num_classes)
        
    # forward pass
    def forward(self,x):
        
        # Pass through first conv layer, batchnorm, relu, and pool
        x = x.float()
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Pass through second conv layer, batchnorm, relu, and pool
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Pass through third conv layer, batchnorm, relu, and pool
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)

        # Update the fully connected layer input size based on the output from CNN
        if self.fc1 is None:  
            input_size = x.size(1)  
            self.fc1 = nn.Linear(input_size, 128)

        # Pass through fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
# Instantiate the model
net = Audio_CNN(num_classes=num_classes, dropout=dropout)

# Loss and Optimizer
optimizer = optim.Adam(net.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# training loop
def train(net, train_loader, epochs, print_every=10):

    counter = 0 
    epoch_train_loss = []
    epoch_val_loss = []
    
    # train for some number of epochs
    net.train()
    for e in range(epochs):
        running_train_loss = 0.0

        # batch loop
        for inputs, labels in train_loader:
            counter += 1

            inputs, labels = inputs.to(device), labels.to(device)

            # zero accumulated gradients
            net.zero_grad()

            # get the output from the model
            output = net(inputs)

            # calculate the loss and perform backprop
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()

            # loss stats
            if counter % print_every == 0:
                # Get validation loss
                val_losses = []
                net.eval()
                running_val_loss = 0.0
                
                for inputs, labels in valid_loader:

                    inputs, labels = inputs.to(device), labels.to(device)

                    output = net(inputs)
                    val_loss = criterion(output, labels)

                    val_losses.append(val_loss.item())
                    running_val_loss += val_loss.item()

                net.train()
                print(f"Epoch: {e+1}/{epochs}...",
                      f"Step: {counter}...",
                      f"Loss: {loss.item()}...",
                      f"Val Loss: {np.mean(val_losses)}")
        
        # Average training loss for the epoch
        avg_train_loss = running_train_loss / len(train_loader)
        epoch_train_loss.append(avg_train_loss)
        
        # Average validation loss for the epoch
        avg_val_loss = running_val_loss / len(valid_loader)
        epoch_val_loss.append(avg_val_loss)
                
    # Save the trained model final checkpoint
    torch.save({
        'epoch': epochs,  
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, 'Audio/audio_CNN_model_checkpoint.pth')
    
    # save train and val loss plot
    plt.plot(epoch_train_loss, label='Training Loss')
    plt.plot(epoch_val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.title('Training and Validation Loss over Epochs')
    plt.savefig('Audio/Audio_loss_plot.png')

# training params
epochs = 100
print_every = 10

train(net, train_loader, epochs, print_every=print_every)

# Get test data loss and accuracy

test_losses = [] # track loss
num_correct = 0

net.eval()
# iterate over test data
for inputs, labels in test_loader:

    inputs, labels = inputs.to(device), labels.to(device)
    
    # get predicted outputs
    output = net(inputs)
    
    # calculate loss
    test_loss = criterion(output, labels)
    test_losses.append(test_loss.item())
    
    # convert output probabilities to predicted class, get max prob class
    pred = torch.argmax(output, dim=1) 
    
    # compare predictions to true label
    correct_tensor = pred.eq(labels)
    correct = np.squeeze(correct_tensor.numpy()) if device == 'cpu' else np.squeeze(correct_tensor.cpu().numpy())
    num_correct += np.sum(correct)

# avg test loss
print(f"Test loss: {np.mean(test_losses)}")

# accuracy over all test data
test_acc = num_correct / len(test_loader.dataset)
print(f"Test accuracy: {test_acc}")