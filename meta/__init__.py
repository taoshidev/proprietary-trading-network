import json


def load_version(file_path):
    try:
        # Open the file and load the JSON data into a dictionary
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError as fnf_error:
        print(f"File not found: {fnf_error}")
        return None
    except json.JSONDecodeError as decode_error:
        print(f"Error decoding JSON: {decode_error}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
