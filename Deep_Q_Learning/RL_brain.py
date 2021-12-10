import numpy as np
import torch
from torch import nn

np.random.seed(1)
torch.manual_seed(1)


class DQNNet(nn.Module):
    def __init__(self, n_actions, n_features):
        super(DQNNet, self).__init__()
        self.out_layer = torch.nn.Sequential(nn.Linear(n_features, 10),
                                             nn.ReLU(),
                                             nn.Linear(10, n_actions)
                                             )

    def forward(self, x):
        return self.out_layer(x)


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
    ):
        self.memory_counter = 0
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        # total learning step
        self.learn_step_counter = 0
        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_net()
        self.optimizer = torch.optim.Adam(self.evaluate_net.parameters(),
                                          learning_rate)
        self.cost_his = []

    def _build_net(self):
        self.evaluate_net = DQNNet(self.n_actions, self.n_features)
        self.target_net = type(self.evaluate_net)(self.n_actions, self.n_features)
        self.target_net.load_state_dict(self.evaluate_net.state_dict())  # copy weights and stuff

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def choose_action(self, observation):
        state = torch.Tensor(observation[np.newaxis, :])

        if np.random.uniform() < self.epsilon:
            actions_value = self.evaluate_net(state)
            action = actions_value.argmax(axis=1).numpy()[0]
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.evaluate_net.state_dict())  # copy weights and stuff

        # sample batch memory from all memory
        indices = np.random.choice(self.memory_size, size=self.batch_size)
        batch_memory = self.memory[indices, :]

        s = torch.Tensor(batch_memory[:, :self.n_features])
        u = torch.LongTensor(batch_memory[:, self.n_features, np.newaxis])
        r = torch.Tensor(batch_memory[:, self.n_features + 1, np.newaxis])
        s_ = torch.Tensor(batch_memory[:, -self.n_features:])

        q_eval = self.evaluate_net(s).gather(1, u)
        q_next = self.target_net(s_).detach().max(axis=1).values
        delta = r + self.gamma * q_next - q_eval

        self.optimizer.zero_grad()
        loss = torch.mean(delta ** 2)
        # train eval network
        loss.backward()
        self.optimizer.step()
        self.cost_his.append(loss.data.numpy())

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()
