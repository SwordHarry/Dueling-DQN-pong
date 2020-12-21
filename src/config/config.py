import os

import torch

# 相关配置和超参数
# 超参数
USE_CUDA = torch.cuda.is_available()
GAMMA = 0.99
EPSILON_MAX = 1
EPSILON_MIN = 0.01
EPS_DECAY = 30000
TRAIN_FRAMES = 1500000
LEARNING_RATE = 2e-4
MAX_BUFF = 100000
UPDATE_TAR_INTERVAL = 1000
BATCH_SIZE = 32
LEARNING_START = 10000

# config for the log or file path
PRINT_INTERVAL = 1000
LOG_INTERVAL = 1000
ENV_ID = 'PongNoFrameskip-v4'  # only use in no frameskip environment
# model save path
TEMP_PATH = os.path.join("model", "temp")
RESULT_PATH = os.path.join("model", "result")
# model name
MODEL_FILE_NAME = "DuelingDQN_dict"
MODEL_FILE_FORMAT = ".pth"

# video save path
VIDEO_PATH = os.path.join("videos", "dqnPongVideo")
# image save path
IMG_PATH = "img"
IMG_FILE = "reward_learning_curve"
IMG_FILE_LR = "reward_learning_curve_lr"
# test model
TEST_FRAMES = 100000
# train log file
TRAIN_LOG_PATH = "log"
TRAIN_LOG_FILE = "train_log.csv"
TRAIN_LOG_FILE_LR = "train_log_lr.csv"
