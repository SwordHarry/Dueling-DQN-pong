import csv
import os
from src.config.config import TRAIN_LOG_PATH, TRAIN_LOG_FILE


# log: output (episode, episode_reward) to data_episode.csv
def log_to_file(data, log_file=TRAIN_LOG_FILE):
    with open(os.path.join(TRAIN_LOG_PATH, log_file), 'a', newline='') as log:
        writer = csv.writer(log)
        writer.writerow(data)


# clear the log
def clear_log(log_file=TRAIN_LOG_FILE):
    with open(os.path.join(TRAIN_LOG_PATH, log_file), 'w', newline='') as log:
        log.truncate()
