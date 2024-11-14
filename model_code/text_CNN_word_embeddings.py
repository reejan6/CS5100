import pandas as pd
import numpy as np
import os
import nltk
from gensim.models import KeyedVectors
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# Text CNN Sentiment Classifier for word embeddings and an embedding layer
class TextCNN(nn.Module):
    def __init__(self, embed_model, embedding_dim, vocab_size,
                 num_filters, num_classes, dropout, kernel_sizes):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(torch.from_numpy(embed_model.vectors))
        self.convs = nn.ModuleList(
            nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embedding_dim), padding = (k - 2,0)) 
            for k in kernel_sizes])
        )
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.sig = nn.Sigmoid()
    
    def conv_and_pool(self, x, conv):
        # fix sizing
        x = F.relu(conv(x)).squeeze(3)
        
        # pool over conv_seq_length
        x_max = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x_max
    
    def forward(self, x):
        # embedded vectors
        embeds = self.embedding(x) 
        embeds = embeds.unsqueeze(1)
        
        # get output of each conv-pool layer
        conv_results = [self.conv_and_pool(embeds, conv) for conv in self.convs]
        
        # concatenate results and add dropout
        x = torch.cat(conv_results, 1)
        x = self.dropout(x)
        
        # final logit
        logit = self.fc(x) 
        
        # sigmoid
        return self.sig(logit)
    
def get_embed_idx(embed_lookup, data_by_words):
    """
    Purpose: get embedding idx for each word in each text
    Args:
        embed_lookup: lookup embedding index mapping
        data_by_words: text data split by word
    Returns: embed index mapping
    """

    embed_idx = []
    for word_doc in data_by_words:
        ints = []
        for word in word_doc:
            try:
                idx = embed_lookup.key_to_index[word]
            except: 
                idx = 0
            ints.append(idx)
        embed_idx.append(ints)
    
    return embed_idx

def pad_features(embed_indexed_texts, seq_length):
    """
    Purpose: to make sure inputs are consistent lengths
    Args:
        embed_indexed_texts: text embedding indices
        seq_length: sequence length
    Returns: index mapping features
    """
    
    # getting the correct rows x cols shape
    features = np.zeros((len(embed_indexed_texts), seq_length), dtype=int)

    for i, row in enumerate(embed_indexed_texts):
        features[i, -len(row):] = np.array(row)[:seq_length]
    
    return features
    
def load_data(filename, word2vec_path, batch_size):
    """
    Purpose: load in data
    Args:
        filename: path to text data
        word2vec_path: path to trained word embeddings
        batch_size: size of batches
    Returns: train, validation, and test split data loaders and vocab size
    """

    nltk.download('punkt_tab')

    # import data
    text_df = pd.read_pickle(filename)

    # tokenize text
    data_by_words = []
    # loop through texts
    for i in text_df['text']:
        # get words, tokenize
        value = nltk.word_tokenize(i)
        data_by_words.append(value)
        
    # encode y labels
    labelencoder = LabelEncoder()
    y = list(text_df['emotions'])
    y = labelencoder.fit_transform(y)

    # get embedding look up table for embedding layer
    embed_lookup = KeyedVectors.load_word2vec_format(word2vec_path, binary = False)
    embed_indexed_texts = get_embed_idx(embed_lookup, data_by_words)
    vocab_size = len(embed_lookup.key_to_index)

    seq_length = 200
    X_features = pad_features(embed_indexed_texts, seq_length)

    # split into train, val, test
    X_train, X_val_test, y_train, y_val_test = train_test_split(X_features, y, test_size=0.2)
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.50)

    # create Tensor datasets
    train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_data = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    # shuffling and batching data
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(val_data, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)
    
    return train_loader, valid_loader, test_loader, vocab_size, embed_lookup

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
    }, os.path.join(save_dir, 'word_embedding_model_checkpoint.pth'))
    
    # save train and val loss plot
    plt.plot(epoch_train_loss, label='Training Loss')
    plt.plot(epoch_val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.title('Training and Validation Loss over Epochs')
    plt.savefig(os.path.join(save_dir, 'Text_loss_word_embedding_plot.png'))
    
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
    test_losses = []
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
    
def run_text_word_embedding_cnn(
    data_path,
    model_path,
    batch_size,
    save_dir,
    dropout = 0.5,
    lr = 0.0001,
    epochs = 100
):
    """
    Purpose: Train and test the model
    Args:
        data_path: path to text data
        model_path: path to word2vec model
        batch_size: size of batches
        save_dir: directory to save trained model weights and biases and loss plot
        dropout: dropout rate
        lr: learning rate
        epochs: train epochs
    Returns: None
    """
    
    train_loader, valid_loader, test_loader, vocab_size, embed_lookup = load_data(data_path, model_path, batch_size)

    # Hyperparameters
    num_classes = 6
    num_filters = 100

    # Instantiate the model
    net = TextCNN(
        embed_model=embed_lookup,
        embedding_dim=100,
        vocab_size=vocab_size,
        num_filters=num_filters,
        num_classes=num_classes,
        dropout=dropout,
        kernel_sizes=[3,4,5]
    )

    # Loss and Optimizer
    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trained_net = train(save_dir, net, train_loader, valid_loader, device,
                        optimizer, criterion, epochs, print_every = 10)
    
    eval(test_loader, trained_net, device, criterion)