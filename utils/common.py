import os
import time
import pickle
import logging

from functools import wraps

subject_names = [
    "FaceTalk_170725_00137_TA",
    "FaceTalk_170728_03272_TA",
    "FaceTalk_170731_00024_TA",
    "FaceTalk_170809_00138_TA",
    "FaceTalk_170811_03274_TA",
    "FaceTalk_170811_03275_TA",
    "FaceTalk_170904_00128_TA",
    "FaceTalk_170904_03276_TA",
    "FaceTalk_170908_03277_TA",
    "FaceTalk_170912_03278_TA",
    "FaceTalk_170913_03279_TA",
    "FaceTalk_170915_00223_TA",
]

sequence_names = [f"sentence{i+1:02}" for i in range(40)]

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
