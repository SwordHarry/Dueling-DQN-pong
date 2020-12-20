from src.config.config import TRAIN_LOG_PATH, VIDEO_PATH, TEMP_PATH, RESULT_PATH, IMG_PATH
from src.dqn.train import train
from src.utils.dir_processor import create_dir
from src.utils.plotTraining import save_plt_reward_frame

if __name__ == '__main__':
    # create_dir([TRAIN_LOG_PATH, VIDEO_PATH, TEMP_PATH, RESULT_PATH, IMG_PATH])
    # train()
    save_plt_reward_frame()
