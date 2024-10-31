import numpy as np
import pickle
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# load train, val, test data from preprocess notebooks

X_train = np.load('text_embeddings_train.npy', allow_pickle=True)
X_val = np.load('text_embeddings_val.npy', allow_pickle=True)
X_test = np.load('text_embeddings_test.npy', allow_pickle=True)

file = open('train_y.pkl','rb')
y_train = pickle.load(file)
file = open('val_y.pkl','rb')
y_val = pickle.load(file)
file = open('test_y.pkl','rb')
y_test = pickle.load(file)

# create Tensor datasets
train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
val_data = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

# set batch size
batch_size = 50

# shuffling and batching data
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
valid_loader = DataLoader(val_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

# Hyperparameters
num_classes = 6
num_filters = 100
dropout = 0.5
lr = 0.0001

# Text CNN Sentiment Classifier for document embedding input
class TextCNN(nn.Module):
    def __init__(self, num_filters, num_classes, dropout):
        super(TextCNN, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(100, num_filters, kernel_size=1),  
            nn.Conv1d(100, num_filters, kernel_size=1),   
            nn.Conv1d(100, num_filters, kernel_size=1),   
        ])
        self.fc = nn.Linear(3 * num_filters, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        
        # shape is (batch_size, embedding_dim, sequence_length)
        x = x.unsqueeze(2)
        
        # apply convolution and ReLU, followed by max pooling
        conv_outputs = [
            F.relu(conv(x)).max(dim=2)[0]
            for conv in self.convs
        ]
        
        # concatenate the output of each convolution layer
        x = torch.cat(conv_outputs, dim=1)
        
        # apply dropout
        x = self.dropout(x)
        
        # fully connected layer for classification
        output = self.fc(x)
        return output

# Instantiate the model
net = TextCNN(
    num_filters=num_filters,
    num_classes=num_classes,
    dropout=dropout
)

# Loss and Optimizer
optimizer = optim.Adam(net.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# training loop
def train(net, train_loader, epochs, print_every=100):

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
    }, 'code/document_embedding_model_checkpoint.pth')
    
    # save train and val loss plot
    plt.plot(epoch_train_loss, label='Training Loss')
    plt.plot(epoch_val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.savefig('code/Text_loss_document_embedding_plot.png')

# training params
epochs = 100
print_every = 100

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