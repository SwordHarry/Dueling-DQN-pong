import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from DQN.DQN import Dueling_DQN
from utils.buffer import Memory_Buffer


class Dueling_DQNAgent:
    def __init__(self, input_shape, action_space=[], USE_CUDA=False, memory_size=10000, epsilon=1, lr=1e-4):
        self.epsilon = epsilon
        self.action_space = action_space
        self.memory_buffer = Memory_Buffer(memory_size)
        self.Dueling_DQN = Dueling_DQN(input_shape=input_shape, num_outputs=action_space.n)
        self.Dueling_DQN_target = Dueling_DQN(input_shape=input_shape, num_outputs=action_space.n)
        self.Dueling_DQN_target.load_state_dict(self.Dueling_DQN.state_dict())

        self.USE_CUDA = USE_CUDA
        if USE_CUDA:
            self.Dueling_DQN = self.Dueling_DQN.cuda()
            self.Dueling_DQN_target = self.Dueling_DQN_target.cuda()
        self.optimizer = optim.RMSprop(self.Dueling_DQN.parameters(), lr=lr, eps=0.001, alpha=0.95)

    def observe(self, lazyframe):
        # from Lazy frame to tensor
        state = torch.from_numpy(lazyframe._force().transpose(2, 0, 1)[None] / 255).float()
        if self.USE_CUDA:
            state = state.cuda()
        return state

    def value(self, state):
        q_values = self.Dueling_DQN(state)
        return q_values

    def act(self, state, epsilon=None, train=False):
        """
        sample actions with epsilon-greedy policy
        recap: with p = epsilon pick random action, else pick action with highest Q(s,a)
        """

        q_values = self.value(state).cpu().detach().numpy()
        if not train:
            return q_values.argmax(1)[0]

        if epsilon is None: epsilon = self.epsilon

        if random.random() < epsilon:
            aciton = random.randrange(self.action_space.n)
        else:
            aciton = q_values.argmax(1)[0]
        return aciton

    def compute_td_loss(self, states, actions, rewards, next_states, is_done, gamma=0.99):
        """ Compute td loss using torch operations only. Use the formula above. """
        actions = torch.tensor(actions).long()  # shape: [batch_size]
        rewards = torch.tensor(rewards, dtype=torch.float)  # shape: [batch_size]
        is_done = torch.tensor(is_done).bool()  # shape: [batch_size]

        if self.USE_CUDA:
            actions = actions.cuda()
            rewards = rewards.cuda()
            is_done = is_done.cuda()

        # get q-values for all actions in current states
        predicted_qvalues = self.Dueling_DQN(states)

        # select q-values for chosen actions
        predicted_qvalues_for_actions = predicted_qvalues[
            range(states.shape[0]), actions
        ]

        # compute q-values for all actions in next states
        predicted_next_qvalues = self.Dueling_DQN_target(next_states)  # YOUR CODE

        # compute V*(next_states) using predicted next q-values
        next_state_values = predicted_next_qvalues.max(-1)[0]  # YOUR CODE

        # compute "target q-values" for loss - it's what's inside square parentheses in the above formula.
        target_qvalues_for_actions = rewards + gamma * next_state_values  # YOUR CODE

        # at the last state we shall use simplified formula: Q(s,a) = r(s,a) since s' doesn't exist
        target_qvalues_for_actions = torch.where(
            is_done, rewards, target_qvalues_for_actions)

        # mean squared error loss to minimize
        # loss = torch.mean((predicted_qvalues_for_actions -
        #                   target_qvalues_for_actions.detach()) ** 2)
        loss = F.smooth_l1_loss(predicted_qvalues_for_actions, target_qvalues_for_actions.detach())

        return loss

    def sample_from_buffer(self, batch_size):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in range(batch_size):
            idx = random.randint(0, self.memory_buffer.size() - 1)
            data = self.memory_buffer.buffer[idx]
            frame, action, reward, next_frame, done = data
            states.append(self.observe(frame))
            actions.append(action)
            rewards.append(reward)
            next_states.append(self.observe(next_frame))
            dones.append(done)
        return torch.cat(states), actions, rewards, torch.cat(next_states), dones

    def learn_from_experience(self, batch_size):
        if self.memory_buffer.size() > batch_size:
            states, actions, rewards, next_states, dones = self.sample_from_buffer(batch_size)
            td_loss = self.compute_td_loss(states, actions, rewards, next_states, dones)
            self.optimizer.zero_grad()
            td_loss.backward()
            for param in self.Dueling_DQN.parameters():
                param.grad.data.clamp_(-1, 1)

            self.optimizer.step()
            return (td_loss.item())
        else:
            return (0)

    def load(self):
        self.Dueling_DQN.load_state_dict(torch.load("trainedModel/Dueling_DQN_dict.pth"))
        print('Model loaded.')
