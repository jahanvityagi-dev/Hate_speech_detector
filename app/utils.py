# app/utils.py

import os
import json

def load_text_files_from_directory(directory_path):
    """
    Loads all .txt files in a directory and returns a list of dictionaries:
    [{'text': ..., 'source_file': ...}, ...]
    """
    documents = []
    for file_name in os.listdir(directory_path):
        if file_name.endswith(".txt"):
            full_path = os.path.join(directory_path, file_name)
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
                documents.append({"text": content, "source_file": file_name})
    return documents

def split_into_chunks(text, chunk_size=300):
    """
    Splits a long string into chunks of specified word size.
    Returns a list of text chunks.
    """
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def save_json(data, file_path):
    """Saves a Python object as a JSON file."""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_json(file_path):
    """Loads and returns JSON data from a file."""
    if not os.path.exists(file_path):
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)
