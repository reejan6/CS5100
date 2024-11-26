## Mental Health Assessment by Speech Emotion Recognition and Text Analysis: A Comparative Approach

### Overview
This project performs a comparative analysis of modeling on text vs audio data to do emotion classification, with the larger goal of drawing conclusions about mental health states. This repo supports modeling of text data and audio files using CNNs. For the text data there are two approaches: 1. training on document embedding representations of various texts 2. training on word embedding mappings using an embedding layer in the CNN. For the audio data we are using a CNN trained on Mel-Frequency Cepstral Coefficients (MFCC) representations of audio files. Our model evaluation reports test loss and accuracy. 

### Features

#### Text Classification

- Data Visualization
    - `visualization_code\text_visualization.py`: generates data exploration visualizations

##### Document Embedding CNN

- Data Preprocessing
    - `preprocessing_notebooks\text_data_preprocess.ipynb`: converts raw text data into a document embedding representations

- Emotion Classification/Sentiment Analysis
    - `model_code\text_CNN_document_embeddings.py`
    - Trains and evaluates a CNN to analyze the sentiment of the input text represented as a document embedding
    - Supports 6 sentiment categories: anger, fear, joy, love, sadness, and surprise
    - Returns the test loss and acuuracy to the user
    - Saves trained model weights and biases
    - Saves a train and validation loss plot

##### Word Embedding Layer CNN

- Data Preprocessing
    - `preprocessing_notebooks\text_data_preprocess.ipynb`: trains a word2vec model to support the CNN embedding layer for the word embedding layer CNN

- Emotion Classification/Sentiment Analysis
    - `model_code\text_CNN_word_embeddings.py`
    - Trains and evaluates a CNN to analyze the sentiment of the input text represented as a mapping to word embeddings (to be recognized by the embedding layer in the CNN)
    - Supports 6 sentiment categories: anger, fear, joy, love, sadness, and surprise
    - Returns the test loss and acuuracy to the user
    - Saves trained model weights and biases
    - Saves a train and validation loss plot

#### Audio Classification

- Data Preprocessing
    - `preprocessing_notebooks\audio_data_preprocess.ipynb`: converts raw audio files into Mel-Frequency Cepstral Coefficients (MFCC) representations
    - `preprocessing_notebooks\audio_preprocess_utils.py`: Util functions for preprocessing audio data
    - `preprocessing_notebooks\featurize_from_wav.py`: featurizes a single wav audio file for inference
    - MFCC is a mathematical method which transforms the power spectrum of an audio signal to a small number of coefficients representing power of the audio signal in a frequency region (a region of pitch) taken with respect to time.
    - Adds white noise to the audio signals, helps to mask the effect of random noise present in the training set, also creates pseudo-new training samples and offsets the impact of noise intrinsic to the dataset.
    - Helps to make the dataset more representative of noisy real-world data

- Data Visualization
    - `visualization_code\audio_visualization.py`: generates data exploration visualizations

- Emotion Classification/Sentiment Analysis
    - `model_code\audio_CNN.py`
    - Trains and evaluates a CNN to analyze the sentiment of the input audio represented as Mel-Frequency Cepstral Coefficients (MFCC)
    - Supports 8 sentiment categories: calm, happy, sad, angry, fearful, surprise, and disgust
    - Returns the test loss and acuuracy to the user
    - Saves trained model weights and biases
    - Saves a train and validation loss plot

### Data Sources

Note: Data too large to store in the repo is stored in this shared Google Drive Folder https://drive.google.com/drive/u/2/folders/1telwNIOOOykOXPYuTyqMKyC1Mwp5VwB9

- Document Embedding CNN
    - Raw Data
        - `data\merged_training.pkl`
    - Preprocessed data, ready for model training
        - `text_embeddings_train.npy` in drive folder
        - `text_embeddings_val.npy` in drive folder
        - `text_embeddings_test.npy` in drive folder
        - `train_y.pkl` in drive folder
        - `val_y.pkl` in drive folder
        - `test_y.pkl` in drive folder
- Word Embedding Layer CNN
    - Raw Data, ready for model training
        - `data\merged_training.pkl`
    - Trained word2vec model for embedding layer
        - `word2vec.model` in drive folder
- Audio CNN
    - Raw Data
        - Downloaded from: https://zenodo.org/records/1188976 
            - Audio Speech Actors Files
    - Preprocessed data, ready for model training
        - `features+labels.npy` in drive folder

### Repo Overview/Structure

- `/config_files`: stores config files to run model training from the command line
    - `audio_cnn_infer_config.json`: example config to infer on a single featurized audio sample (converted into MFCC already)
    - `audio_cnn_train_config.json`: example config file to run the audio cnn train and eval
    - `text_doc_embed_infer_config.json`: example config file to infer on a single document/ piece of text using document embeddings
    - `text_doc_embed_cnn_train_config.json`: example config file to run the text document embedding cnn train and eval
    - `text_word_embed_infer_config.json`: example config file to infer on a single document/ piece of text using word embeddings
    - `text_word_embed_cnn_train_config.json`: example config file to run the word embedding layer cnn train and eval
- `/data`: data used in model training (small enough size to store in repo)
    - `merged_training.pkl`: raw text data used in word embedding layer cnn
- `/model_checkpoint`: trained model weights and biases, can be used to load models and infer
    - `audio_CNN_model_checkpoint.pth`: audio cnn model
    - `document_embedding_model_checkpoint.pth`: text document embedding cnn model
    - `word_embedding_model_checkpoint.pth`: word embedding layer cnn model
- `/model_code`: CNN and train and eval code
    - `audio_CNN.py`: audio cnn code
    - `text_CNN_document_embeddings.py`: text document embedding cnn code
    - `text_CNN_word_embeddings.py`: text word embedding layer cnn code
- `/plots`: loss plots from model training
    - `Audio_loss_plot.png`
    - `Text_loss_document_embeding_plot.png`
    - `Text_loss_word_embedding_plot.png`
- `/preprocessing_notebooks`: notebooks for preprocessing data used to train and infer on these models
    - `audio_data_preprocess.ipynb`: preprocesses the RAVDESS dataset
    - `audio_preprocess_utils`: utils functions for preprocessing audio data into MFCC
    - `featurize_from_wav.py`: preprocesses a single .wav audio file to infer on
    - `text_data_preprocess.ipynb`: preprocesses the text dataset CARER
- `/visualization_code`: data exploration visualization code
    - `audio_visualization.py`
    - `text_visualization.py`
- `.gitignore`
- `project_flow_chart.pdf`: project flow chart pdf, data and model pipeline
- `Results.csv`: model training results with different hyperparameters
- `rnn_models.py`: code to run models from config file

### Getting Started

#### Prerequisites

Ensure you have all necessary data files downloaded as listed in the Data Sources section of the ReadMe

Ensure you have the following installed to run our models and preprocessing steps
- python
- torch
- numpy
- matplotlib
- pickle
- nltk
- gensim
- pandas
- sklearn
- librosa
- seaborn
- collections

#### Config Files

Train Params:
- run_mode: train or infer
- preprocessed_data_paths: List of paths to preprocessed data needed to run the model. (Expected structure per model in example configs)
- save_dir: path to directory to save model results (trained model weights and biases as well as loss plot)
- model_type: model to run (audio, text_doc, text_word)
- batch_size
- learning rate
- epochs: train epochs
- dropout: dropout rate
- word2vec_path: path to trained word2vec model for text word embedding layer cnn (null for audio cnn and document embedding cnn)

Infer Params:
- run_mode: train or infer
- model_type: model to infer with (audio, text_doc, text_word)
- net_path: path to the model weights and biases saved from model training (pth file)
- input: input to infer on. A string text for text models and a .npy file returned from running `featurize_from_wav.py` on a .wav audio file
- dropout: dropout rate
- word2vec_path: path to trained word2vec model for text cnns (null for audio cnn)

#### Running model training and evaluation or inference

1. clone the repo `git clone https://github.com/reejan6/CS5100.git`
2. Download all necessary data
3. Update the config files in `/config_files` to point to the correct data paths
4. navigate to the `/model_code` folder. `cd model_code`
5. Run the following command: `python run_models.py --config <path to config file>` (either inference or train config depending on goal)

Examples:

`python run_models.py --config "CS5100/config_files/audio_cnn_train_config.json"`

`python run_models.py --config "CS5100/config_files/audio_cnn_infer_config.json"`

`python run_models.py --config "CS5100/config_files/text_doc_embed_cnn_train_config.json"`

`python run_models.py --config "CS5100/config_files/text_doc_embed_cnn_infer_config.json"`

`python run_models.py --config "CS5100/config_files/text_word_embed_cnn_train_config.json"`

`python run_models.py --config "CS5100/config_files/text_word_embed_cnn_infer_config.json"`

### Resources/ References

RAVDESS Dataset: https://zenodo.org/records/1188976

CARER Dataset: https://paperswithcode.com/dataset/emotion

RAVDESS Preprocessing Steps: https://nbviewer.org/github/IliaZenkov/transformer_cnn_parallel_audio_classification/blob/main/notebooks/Parallel_is_All_You_Want.ipynb#Load-Data-and-Extract-Features