import csv
import os

import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.pyplot import MultipleLocator

from src.config.config import TRAIN_LOG_PATH, TRAIN_LOG_FILE, IMG_PATH


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def plot_training(episodes, rewards, is_save=False):
    clear_output(True)
    # title and label
    plt.title('reward learning curve', fontsize=24)
    plt.xlabel("episode")
    plt.ylabel("mean reward")

    # 坐标刻度设置
    x_major_locator = MultipleLocator(100)
    y_major_locator = MultipleLocator(10)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax.yaxis.set_major_locator(y_major_locator)
    # set the plot
    plt.plot(episodes, rewards)
    if is_save:
        plt.savefig(IMG_PATH)
    plt.show()


# output to plt
def save_plt_reward_frame():
    episode_list = []
    reward_list = []
    with open(os.path.join(TRAIN_LOG_PATH, TRAIN_LOG_FILE)) as csv_file:
        csv_reader = csv.reader(csv_file)  # 使用csv.reader读取csvfile中的文件
        title_list = next(csv_reader)  # 读取第一行每一列的标题，或者略过第一行有问题的数据
        for row in csv_reader:  # 将csv 文件中的数据保存到birth_data中
            episode = row[1]
            if len(episode_list) > 0 and episode_list[-1] == episode:
                continue
            reward = row[2]
            episode_list.append(episode)
            reward_list.append(reward)
    plot_training(episode_list, reward_list)
