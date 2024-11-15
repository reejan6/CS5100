import argparse
import json
import os
from model_code.audio_CNN import run_audio_cnn
from model_code.text_CNN_document_embeddings import run_text_doc_embedding_cnn
from model_code.text_CNN_word_embeddings import run_text_word_embedding_cnn
from model_code.text_CNN_word_embeddings import run_text_word_embedding_infer
from model_code.text_CNN_document_embeddings import run_text_doc_embedding_infer

def load_config(config_path):
    """Load JSON configuration file."""
    if os.path.exists(config_path):
        with open(config_path, 'r') as file:
            config = json.load(file)
        return config
    else:
        raise FileNotFoundError(f"Config file not found at: {config_path}")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run script with a JSON configuration file.")
    parser.add_argument(
        '--config', 
        type=str, 
        required=True, 
        help='Path to the JSON configuration file'
    )
    args = parser.parse_args()

    # Load the configuration file
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(e)
        return
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON config file: {e}")
        return
    
    try:
        run_mode = config["run_mode"]
    except KeyError as e:
        print(f"Missing key in configuration: {e}")
        return
    
    if run_mode == "train":
        try:
            preprocessed_data_paths = config["preprocessed_data_paths"]
            save_dir = config["save_dir"]
            model_type = config["model_type"]
            batch_size = config["batch_size"]
            learning_rate = config["learning_rate"]
            epochs = config["epochs"]
            dropout = config["dropout"]
            word2vec_path = config["word2vec_path"]
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

    else:
        try:
            net_path = config["net_path"]
            input_text = config["input_text"]
            word2vec_path = config["word2vec_path"]
            dropout = config["dropout"]
            model_type = config["model_type"]
        except KeyError as e:
            print(f"Missing key in configuration: {e}")
            return
        
        if model_type == 'text_word':
            run_text_word_embedding_infer(
                net_path,
                input_text,
                word2vec_path,
                dropout
            )
        
        elif model_type == "text_doc":
            run_text_doc_embedding_infer(
                net_path,
                input_text,
                word2vec_path,
                dropout
            )
        
        else:
            print("Inference not implemented for other models.")   

if __name__ == '__main__':
    main()