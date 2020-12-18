import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from DQN.agent import Dueling_DQNAgent
from utils.atari_wrappers import make_atari, wrap_deepmind
from tensorboardX import SummaryWriter

# Training DQN in PongNoFrameskip-v4
from DQN.test import test

env = make_atari('PongNoFrameskip-v4')
env = wrap_deepmind(env, scale=False, frame_stack=True)

# 超参数
gamma = 0.99
epsilon_max = 1
epsilon_min = 0.01
eps_decay = 30000
frames = 1000000
USE_CUDA = torch.cuda.is_available()
learning_rate = 2e-4
max_buff = 100000
update_tar_interval = 1000
batch_size = 32
print_interval = 1000
log_interval = 1000
learning_start = 10000
win_reward = 18  # Pong-v4
win_break = True


# train the model
def train():
    frame = env.reset()
    action_space = env.action_space
    # action_dim = env.action_space.n
    # state_dim = env.observation_space.shape[0]
    # state_channel = env.observation_space.shape[2]
    input_shape = frame._force().transpose(2, 0, 1).shape
    agent = Dueling_DQNAgent(input_shape=input_shape, action_space=action_space, USE_CUDA=USE_CUDA, lr=learning_rate)

    episode_reward = 0
    all_rewards = []
    losses = []
    episode_num = 0
    # tensorboard
    summary_writer = SummaryWriter(log_dir="Dueling_DQN", comment="good_makeatari")

    # e-greedy decay
    epsilon_by_frame = lambda frame_idx: epsilon_min + (epsilon_max - epsilon_min) * math.exp(
        -1. * frame_idx / eps_decay)
    # plt.plot([epsilon_by_frame(i) for i in range(10000)])

    for i in range(frames):
        epsilon = epsilon_by_frame(i)
        state_tensor = agent.observe(frame)
        action = agent.act(state_tensor, epsilon, train=True)

        next_frame, reward, done, _ = env.step(action)

        episode_reward += reward
        agent.memory_buffer.push(frame, action, reward, next_frame, done)
        frame = next_frame

        loss = 0
        if agent.memory_buffer.size() >= learning_start:
            loss = agent.learn_from_experience(batch_size)
            losses.append(loss)

        if i % print_interval == 0:
            print("frames: %5d, reward: %5f, loss: %4f, epsilon: %5f, episode: %4d" % (
                i, np.mean(all_rewards[-10:]), loss, epsilon, episode_num))
            summary_writer.add_scalar("Temporal Difference Loss", loss, i)
            summary_writer.add_scalar("Mean Reward", np.mean(all_rewards[-10:]), i)
            summary_writer.add_scalar("Epsilon", epsilon, i)
            torch.save(agent.Dueling_DQN.state_dict(), "trainedModel/Dueling_DQN_dict.pth")
        if i % update_tar_interval == 0:
            agent.Dueling_DQN_target.load_state_dict(agent.Dueling_DQN.state_dict())

        if done:
            frame = env.reset()

            all_rewards.append(episode_reward)
            episode_reward = 0
            episode_num += 1
            avg_reward = float(np.mean(all_rewards[-100:]))
    torch.save(agent.Dueling_DQN.state_dict(), "trainedModel/Dueling_DQN_dict.pth")
    summary_writer.close()


# Create and wrap the environment
def create_and_wrap_env():
    env = make_atari('PongNoFrameskip-v4')  # only use in no frameskip environment
    env = wrap_deepmind(env, scale=False, frame_stack=True)
    # env.render()
    test = env.reset()
    for i in range(100):
        test = env.step(env.action_space.sample())[0]

    plt.imshow(test._force()[..., 0])


if __name__ == '__main__':
    is_train = False
    if is_train:
        train()
    else:
        test()
