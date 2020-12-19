from src.config.config import IS_TRAIN
from src.dqn.test import test
from src.dqn.train import train


if __name__ == '__main__':
    if IS_TRAIN:
        train()
    else:
        test()
