import csv
import math
import os
import torch
import numpy as np
from src.dqn.agent import DuelingDQNAgent
from src.config.config import EPSILON_MAX, EPSILON_MIN, EPS_DECAY, TRAIN_FRAMES, LEARNING_START, BATCH_SIZE, \
    PRINT_INTERVAL, \
    UPDATE_TAR_INTERVAL, USE_CUDA, LEARNING_RATE, ENV_ID, TEMP_PATH, RESULT_PATH, \
    MODEL_FILE_FORMAT, MODEL_FILE_NAME, TRAIN_LOG_PATH, TRAIN_LOG_FILE
from src.utils.atari_wrappers import make_atari, wrap_deepmind


def epsilon_by_frame(frame_idx):
    return EPSILON_MIN + (EPSILON_MAX - EPSILON_MIN) * math.exp(
        -1. * frame_idx / EPS_DECAY)


# train the model
def train():
    # create the env
    env = make_atari(ENV_ID)
    env = wrap_deepmind(env)
    frame = env.reset()

    # create the dqn agent
    action_space = env.action_space
    input_shape = frame._force().transpose(2, 0, 1).shape
    agent = DuelingDQNAgent(input_shape=input_shape, action_space=action_space, use_cuda=USE_CUDA, lr=LEARNING_RATE)

    # begin the train
    episode_reward = 0
    all_rewards = []
    losses = []
    episode_num = 0

    for i in range(TRAIN_FRAMES):
        epsilon = epsilon_by_frame(i)
        state_tensor = agent.observe(frame)
        action = agent.act(state_tensor, epsilon, train=True)
        next_frame, reward, done, _ = env.step(action)
        episode_reward += reward
        agent.memory_buffer.push(frame, action, reward, next_frame, done)
        frame = next_frame
        loss = 0
        if agent.memory_buffer.size() >= LEARNING_START:
            loss = agent.learn_from_experience(BATCH_SIZE)
            losses.append(loss)
        # print the log and save the temp model
        if i % PRINT_INTERVAL == 0:
            data_e_er = (i, episode_num, np.mean(all_rewards[-10:]), loss, epsilon)
            # log: output (episode, episode_reward) to data_episode.csv
            with open(os.path.join(TRAIN_LOG_PATH, TRAIN_LOG_FILE), 'a', newline='') as log:
                writer = csv.writer(log)
                writer.writerow(data_e_er)
            print("FRAMES: %5d, reward: %5f, loss: %4f, epsilon: %5f, episode: %4d" % (
                i, np.mean(all_rewards[-10:]), loss, epsilon, episode_num))
            # save the temp model
            temp_pth_path = os.path.join(TEMP_PATH,
                                         MODEL_FILE_NAME + str(i) + MODEL_FILE_FORMAT)
            torch.save(agent.DuelingDQN.state_dict(), temp_pth_path)
        if i % UPDATE_TAR_INTERVAL == 0:
            agent.DuelingDQN_target.load_state_dict(agent.DuelingDQN.state_dict())

        if done:
            frame = env.reset()
            all_rewards.append(episode_reward)
            episode_reward = 0
            episode_num += 1
            # avg_reward = float(np.mean(all_rewards[-100:]))
    # save the final result model
    final_pth_path = os.path.join(RESULT_PATH, MODEL_FILE_NAME + MODEL_FILE_FORMAT)
    torch.save(agent.DuelingDQN.state_dict(), final_pth_path)
