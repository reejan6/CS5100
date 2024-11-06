import argparse
import configparser
import os

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

    # Example of reading from the configuration
    try:
        setting1 = config.get('SectionName', 'setting1')
        setting2 = config.getint('SectionName', 'setting2')
        print(f"Setting1: {setting1}")
        print(f"Setting2: {setting2}")
    except KeyError as e:
        print(f"Missing key in configuration: {e}")
        return

    # Additional script logic can follow here...
    print("Script is running with the configuration!")

if __name__ == '__main__':
    main()