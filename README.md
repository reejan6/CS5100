## Mental Health Assessment by Speech Emotion Recognition and Text Analysis: A Comparative Approach

### Overview
This project performs a comparative analysis of modeling on text vs audio data to do emotion classification, with the larger goal of drawing conclusions about mental health states. This repo supports modeling of text data and audio files using CNNs. For the text data there are two approaches: 1. training on document embedding representations of various texts 2. training on word embedding mappings using an embedding layer in the CNN. For the audio data we are using a CNN trained on Mel-Frequency Cepstral Coefficients (MFCC) representations of audio files. Our model evaluation reports test loss and accuracy. 

### Features

#### Text Classification

- Data Visualization
    - In `visualization_code\text_visualization.py`, generates helpful data exploratory visualizations

##### Document Embedding CNN

- Data Preprocessing
    - In `preprocessing_notebooks\text_data_preprocess.ipynb`, converts raw text data into a document embedding representations

- Emotion Classification/Sentiment Analysis
    - Trains and evaluates a CNN to analyze the sentiment of the input text represented as a document embedding
    - Supports 6 sentiment categories: anger, fear, joy, love, sadness, and surprise
    - Returns the test loss and acuuracy to the user

##### Word Embedding Layer CNN

- Data Preprocessing
    - Trains a word2vec model to support the CNN embedding layer for the word embedding layer CNN

- Emotion Classification/Sentiment Analysis
    - Trains and evaluates a CNN to analyze the sentiment of the input text represented as a mapping to word embeddings (to be recognized by the embedding layer in the CNN)
    - Supports 6 sentiment categories: anger, fear, joy, love, sadness, and surprise
    - Returns the test loss and acuuracy to the user

#### Audio Classification

- Data Preprocessing
    - In `preprocessing_notebooks\audio_data_preprocess.ipynb`, converts raw audio files into Mel-Frequency Cepstral Coefficients (MFCC) representations
    - MFCC is a mathematical method which transforms the power spectrum of an audio signal to a small number of coefficients representing power of the audio signal in a frequency region (a region of pitch) taken with respect to time.
    - Adds white noise to the audio signals, helps to mask the effect of random noise present in the training set, also creates pseudo-new training samples and offsets the impact of noise intrinsic to the dataset.
    - Helps to make the dataset more representative of noisy real-world data

- Data Visualization
    - In `visualization_code\audio_visualization.py`, generates helpful data exploratory visualizations

- Emotion Classification/Sentiment Analysis
    - Trains and evaluates a CNN to analyze the sentiment of the input audio represented as Mel-Frequency Cepstral Coefficients (MFCC)
    - Supports 8 sentiment categories: calm, happy, sad, angry, fearful, surprise, and disgust
    - Returns the test loss and acuuracy to the user

### Repo Overview/Structure

- 

### Getting Started



python run_models.py --config "C:/Users/jenni/virtualEnv/CS 5100/Final Project/project_code/CS5100/config_files/audio_cnn_train_config.json"

python run_models.py --config "C:/Users/jenni/virtualEnv/CS 5100/Final Project/project_code/CS5100/config_files/text_doc_embed_cnn_train_config.json"

python run_models.py --config "C:/Users/jenni/virtualEnv/CS 5100/Final Project/project_code/CS5100/config_files/text_word_embed_cnn_train_config.json"
