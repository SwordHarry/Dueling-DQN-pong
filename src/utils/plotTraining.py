import csv
import os

import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.pyplot import MultipleLocator

from src.config.config import TRAIN_LOG_PATH, TRAIN_LOG_FILE, IMG_PATH, TRAIN_LOG_FILE_LR


# show or save tht plt of the train
def plot_training(episodes_list, rewards_list, is_save=True, is_compare=True):
    clear_output(True)
    # title and label
    plt.title('reward learning curve', fontsize=24)
    plt.xlabel("episode(frames/k)")
    plt.ylabel("mean reward")

    # 坐标刻度设置
    x_major_locator = MultipleLocator(100)
    y_major_locator = MultipleLocator(10)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax.yaxis.set_major_locator(y_major_locator)

    # fig, ax = plt.subplots()
    # set the plot
    plt.plot(episodes_list[0], rewards_list[0], color='blue', label="origin")
    if is_compare:
        plt.plot(episodes_list[1], rewards_list[1], color="red", label="lr")
    plt.legend()
    if is_save:
        plt.savefig(IMG_PATH)
    plt.show()


# output to plt
def save_plt_reward_frame():
    episodes_list = []
    rewards_list = []
    e, r = get_episodes_rewards(TRAIN_LOG_FILE)
    episodes_list.append(e)
    rewards_list.append(r)
    e_lr, r_lr = get_episodes_rewards(TRAIN_LOG_FILE_LR)
    episodes_list.append(e_lr)
    rewards_list.append(r_lr)
    plot_training(episodes_list, rewards_list)


def get_episodes_rewards(file_str):
    episode_list = []
    reward_list = []
    with open(os.path.join(TRAIN_LOG_PATH, file_str)) as csv_file:
        csv_reader = csv.reader(csv_file)  # 使用csv.reader读取csvfile中的文件
        title_list = next(csv_reader)  # 读取第一行每一列的标题，或者略过第一行有问题的数据
        for row in csv_reader:  # 将csv 文件中的数据保存到birth_data中
            episode = row[0]
            episode = int(episode)/1000
            reward = row[2]
            episode_list.append(episode)
            reward_list.append(reward)
    return list(map(int, episode_list)), list(map(float, reward_list))

