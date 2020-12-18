from DQN.agent import Dueling_DQNAgent
from utils.atari_wrappers import *


def make_atari(env_id):
    env = gym.make(env_id)
    assert 'NoFrameskip' in env.spec.id
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    return env


def test(is_render=False):
    env = make_atari('PongNoFrameskip-v4')
    env = wrap_deepmind(env, scale=False, frame_stack=True)
    frame = env.reset()
    input_shape = frame._force().transpose(2, 0, 1).shape
    # save the videos
    env = gym.wrappers.Monitor(env, './videos/' + 'dqn_pong_video')
    action_space = env.action_space
    agent = Dueling_DQNAgent(input_shape=input_shape, action_space=action_space)
    agent.load()
    total_reward = 0
    frame = env.reset()
    if is_render:
        env.render()
    for _ in range(100000):
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
