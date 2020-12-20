from src.config.config import TRAIN_LOG_PATH, VIDEO_PATH, TEMP_PATH, RESULT_PATH, IMG_PATH
from src.dqn.test import test
from src.utils.dir_processor import create_dir

if __name__ == '__main__':
    create_dir([TRAIN_LOG_PATH, VIDEO_PATH, TEMP_PATH, RESULT_PATH, IMG_PATH])
    test()
