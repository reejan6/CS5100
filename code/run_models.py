import argparse
import configparser
import os
from code.audio_CNN import run_audio_cnn
from code.text_CNN_document_embeddings import run_text_doc_embedding_cnn
from code.text_CNN_word_embeddings import run_text_word_embedding_cnn

def load_config(config_path):
    """Load configuration file using configparser."""
    config = configparser.ConfigParser()
    if os.path.exists(config_path):
        config.read(config_path)
        return config
    else:
        raise FileNotFoundError(f"Config file not found at: {config_path}")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run script with a configuration file.")
    parser.add_argument(
        '--config', 
        type=str, 
        required=True, 
        help='Path to the configuration file'
    )
    args = parser.parse_args()

    # Load the configuration file
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(e)
        return

    try:
        preprocessed_data_paths = config.get("preprocessed_data_paths")
        save_dir = config.get("save_dir")
        model_type = config.get("model_type")
        batch_size = config.get("batch_size")
        learning_rate = config.get("learning_rate")
        epochs = config.get("epochs")
        dropout = config.get("dropout")
        word2vec_path = config.get("word2vec_path")
    except KeyError as e:
        print(f"Missing key in configuration: {e}")
        return
    
    if model_type == 'audio':
        run_audio_cnn(
            preprocessed_data_paths[0],
            batch_size,
            save_dir,
            dropout,
            learning_rate,
            epochs
        )
        
    elif model_type == 'text_doc':
        run_text_doc_embedding_cnn(
            preprocessed_data_paths[0],
            preprocessed_data_paths[1],
            preprocessed_data_paths[2],
            preprocessed_data_paths[3],
            preprocessed_data_paths[4],
            preprocessed_data_paths[5],
            batch_size,
            save_dir,
            dropout,
            learning_rate,
            epochs
        )
        
    elif model_type == 'text_word':
        run_text_word_embedding_cnn(
            preprocessed_data_paths[0],
            word2vec_path,
            batch_size,
            save_dir,
            dropout,
            learning_rate,
            epochs
        )

if __name__ == '__main__':
    main()