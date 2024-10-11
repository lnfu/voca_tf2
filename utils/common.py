import os
import logging


def check_file_exists(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file at {file_path} does not exist.")


def check_and_create_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        logging.info(f"Directory created: {dir_path}")

# TODO
# pickle.load(open(filepath, "rb"), encoding="latin1")