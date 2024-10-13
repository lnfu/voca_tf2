import os
import time
import pickle
import logging

from functools import wraps


def check_file_exists(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file at {file_path} does not exist.")


def check_and_create_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        logging.info(f"Directory created: {dir_path}")


def load_pickle(filepath: str):
    with open(filepath, "rb") as file:
        return pickle.load(file, encoding="latin1")


def log_execution(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logging.info("")
        logging.info(f"開始執行: {func.__name__}")
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            logging.error(f"執行 {func.__name__} 時發生錯誤: {e}")
            raise
        finally:
            end_time = time.time()
            logging.info(f"完成執行: {func.__name__}，耗時 {end_time - start_time:.2f} 秒")
            logging.info("")

    return wrapper
