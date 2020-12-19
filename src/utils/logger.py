import csv
import os
from src.config.config import TRAIN_LOG_PATH, TRAIN_LOG_FILE


def log_to_file(data):
    # log: output (episode, episode_reward) to data_episode.csv
    with open(os.path.join(TRAIN_LOG_PATH, TRAIN_LOG_FILE), 'a', newline='') as log:
        writer = csv.writer(log)
        writer.writerow(data)
