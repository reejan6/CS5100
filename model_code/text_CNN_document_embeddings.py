import numpy as np
import os
import nltk
from gensim.models import Word2Vec
import pickle
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt  

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
        
        # fix shape
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

def load_data(train_X_file, train_y_file, val_X_file, val_y_file, test_X_file, test_y_file, batch_size):
    """
    Purpose: load in preprocessed data
    Args:
        train_X_file: preprocessed train data
        train_y_file: train labels
        val_X_file: val data
        val_y_file: val labels
        test_X_file: test data
        test_y_file: test labels
        batch_size: size of batches
    Returns: train, validation, and test split data loaders
    """
    
    # load train, val, test data from preprocess notebooks

    X_train = np.load(train_X_file, allow_pickle=True)
    X_val = np.load(val_X_file, allow_pickle=True)
    X_test = np.load(test_X_file, allow_pickle=True)

    file = open(train_y_file,'rb')
    y_train = pickle.load(file)
    file = open(val_y_file,'rb')
    y_val = pickle.load(file)
    file = open(test_y_file,'rb')
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
    
    return train_loader, valid_loader, test_loader

def train(
    save_dir,
    net,
    train_loader,
    valid_loader,
    device,
    optimizer,
    criterion,
    epochs = 100,
    print_every=10
):
    """
    Purpose: Train the model
    Args:
        save_dir: directory to save trained model weights and biases and loss plot
        net: model to train
        train_loader: train data loader
        valid_loader: validation data loader
        device: device to run model on
        optimizer: optimizer to use
        criterion: loss function to use
        epochs: training epoch count
        print_every: print loss after number of batches
    Returns: trained model
    """

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
    }, os.path.join(save_dir, 'document_embedding_model_checkpoint.pth'))
    
    # save train and val loss plot
    plt.plot(epoch_train_loss, label='Training Loss')
    plt.plot(epoch_val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.title('Training and Validation Loss over Epochs')
    plt.savefig(os.path.join(save_dir, 'Text_loss_document_embedding_plot.png'))
    
    return net
    
def eval(
    test_loader,
    net,
    device,
    criterion
):
    """
    Purpose: Evaluate/test the model
    Args:
        test_loader: test data loader
        net: trained model to evaluate
        device: device to run eval on
        criterion: loss function
    Returns: None
    """
    
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
    
def run_text_doc_embedding_cnn(
    train_X_file,
    train_y_file,
    val_X_file,
    val_y_file,
    test_X_file,
    test_y_file,
    batch_size,
    save_dir,
    dropout = 0.5,
    lr = 0.0001,
    epochs = 100
):
    """
    Purpose: Train and test the model
    Args:
        train_X_file: preprocessed train data
        train_y_file: train labels
        val_X_file: val data
        val_y_file: val labels
        test_X_file: test data
        test_y_file: test labels
        batch_size: size of batches
        save_dir: directory to save trained model weights and biases and loss plot
        dropout: dropout rate
        lr: learning rate
        epochs: train epochs
    Returns: None
    """
    train_loader, valid_loader, test_loader = load_data(
        train_X_file, train_y_file, val_X_file, val_y_file, test_X_file, test_y_file, batch_size
    )

    # define parameters
    num_classes = 6
    num_filters = 100

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
    
    trained_net = train(save_dir, net, train_loader, valid_loader, device,
                        optimizer, criterion, epochs, print_every = 10)
    
    eval(test_loader, trained_net, device, criterion)
    
def prep_infer_text(
    input_text,
    word2vec_path
):
    """
    Purpose: prep input text for doc embedding model
    Args:
        input_text: text to classify
        word2vec_path: word2vec model 

    Returns: document embedding representation of input text
    """
    tokenized_input = nltk.word_tokenize(input_text)
    model = Word2Vec.load(word2vec_path)
    
    text_embeds = []
    for j in tokenized_input:
        if j in model.wv.key_to_index:
            text_embeds.append(model.wv[j])
    if len(text_embeds) != 0:
        text_embeds = np.mean(text_embeds, axis = 0)
    else:
        print("empty input")
        
    return np.array(text_embeds)

def run_text_word_embedding_infer(
    net_path, input_text, word2vec_path, dropout
):
    """
    Purpose: classify given text input
    Args:
        net_path: path to trained model pth file
        input_text: text to classify
        word2vec_path: path to trained word embeddings
    Returns: classification of input text
    """
    embeds_input = prep_infer_text(input_text, word2vec_path)
    
    net = TextCNN(
        num_filters = 100,
        num_classes = 6,
        dropout = dropout
    )
    checkpoint = torch.load(net_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()

    feature_tensor = torch.from_numpy(embeds_input)
    output = net(feature_tensor)
    pred = torch.argmax(output, dim=1) 
    
    mapping_dict = {
        0: 'anger',
        1: 'fear',
        2: 'joy',
        3: 'love',
        4: 'sadness',
        5: 'surprise'
    }
    
    print("Sentiment Detected: " + mapping_dict[pred.item()])
    