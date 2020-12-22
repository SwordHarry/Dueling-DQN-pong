import csv
import os

from IPython.display import clear_output
import matplotlib.pyplot as plt

from src.config.config import TRAIN_LOG_PATH, TRAIN_LOG_FILE, IMG_PATH, TRAIN_LOG_FILE_LR, IMG_FILE, IMG_FILE_LR


# show or save tht plt of the train
def plot_training(episodes_list, rewards_list, show_index=0, is_save=False, is_compare=True):
    clear_output(True)
    # title and label
    plt.title('reward learning curve', fontsize=24)
    plt.xlabel("episode")
    plt.ylabel("mean reward")

    # 坐标刻度设置
    plt.xlim(0, episodes_list[0][-1] * 1.2)
    plt.ylim(-25, 25)
    # set the plot
    config_list = [('blue', 'origin', IMG_FILE), ('orange', 'batch_size', IMG_FILE_LR)]
    plt.plot(episodes_list[show_index], rewards_list[show_index],
             color=config_list[show_index][0], label=config_list[show_index][1])
    compare_str = ""
    if is_compare:
        plt.plot(episodes_list[1 - show_index], rewards_list[1 - show_index],
                 color=config_list[1 - show_index][0], label=config_list[1 - show_index][1])
        compare_str = "_compare"
    plt.legend()
    if is_save:
        plt.savefig(os.path.join(IMG_PATH, config_list[show_index][2] + compare_str))
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
    plot_training(episodes_list, rewards_list, show_index=0, is_save=True, is_compare=True)


def get_episodes_rewards(file_str):
    episode_list = []
    reward_list = []
    with open(os.path.join(TRAIN_LOG_PATH, file_str)) as csv_file:
        csv_reader = csv.reader(csv_file)  # 使用csv.reader读取csvfile中的文件
        title_list = next(csv_reader)  # 读取第一行每一列的标题，或者略过第一行有问题的数据
        for row in csv_reader:  # 将csv 文件中的数据保存到birth_data中
            episode = row[0]
            reward = row[2]
            episode_list.append(episode)
            reward_list.append(reward)
    return list(map(int, episode_list)), list(map(float, reward_list))
