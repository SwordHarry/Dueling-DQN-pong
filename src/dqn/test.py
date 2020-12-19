import os

from src.config.config import ENV_ID, VIDEO_PATH, TEST_FRAMES
from src.dqn.agent import DuelingDQNAgent
from src.utils.atari_wrappers import *
import matplotlib.pyplot as plt


def make_atari(env_id):
    env = gym.make(env_id)
    assert 'NoFrameskip' in env.spec.id
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    return env


# Create and wrap the environment
def create_and_wrap_env(is_render=False):
    env = make_atari(ENV_ID)  # only use in no frameskip environment
    env = wrap_deepmind(env)
    test_frame = env.reset()
    if is_render:
        env.render()
    for i in range(100):
        test_frame = env.step(env.action_space.sample())[0]
    plt.imshow(test_frame._force()[..., 0])


# test the model
def test(is_render=False):
    # create the env
    env = make_atari(ENV_ID)
    env = wrap_deepmind(env)
    # get the input shape
    frame = env.reset()
    input_shape = frame._force().transpose(2, 0, 1).shape

    # save the videos
    env = gym.wrappers.Monitor(env, VIDEO_PATH, force=True)

    # use the dqn agent to load model
    action_space = env.action_space
    agent = DuelingDQNAgent(input_shape=input_shape, action_space=action_space)
    agent.load()

    # begin test pong
    total_reward = 0
    if is_render:
        env.render()
    # have to reset the initial step
    env.reset()
    for _ in range(TEST_FRAMES):
        state_tensor = agent.observe(frame)
        action = agent.act(state_tensor, train=False)
        next_frame, reward, done, _ = env.step(action)
        total_reward += reward
        frame = next_frame
        if is_render:
            env.render()
        if done:
            print(f'total_reward: {total_reward}')
            env.close()
            break
