import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

np.random.seed(1)
torch.manual_seed(1)


class PolicyNet(nn.Module):
    def __init__(self, n_actions, n_features):
        super(PolicyNet, self).__init__()
        hidden_units = 10
        self.fc_layer = nn.Sequential(nn.Linear(n_features, hidden_units),
                                      nn.Sigmoid(),
                                      nn.Linear(hidden_units, n_actions),
                                      nn.Softmax(dim=-1))

    def forward(self, x):
        output = self.fc_layer(x)
        return output


class PolicyGradient:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.95
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        self.cost_his = []
        self._build_net()

    def _build_net(self):
        self.policy_net = PolicyNet(self.n_actions, self.n_features)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(),
                                          self.lr)

    def choose_action(self, observation):
        state = torch.Tensor(observation[np.newaxis, :])
        prob_weights = self.policy_net(state)
        action_idx = prob_weights.reshape(-1, ).multinomial(num_samples=1).numpy()[0]
        return action_idx

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        # discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        # train on episode
        torch_obs = torch.Tensor(np.vstack(self.ep_obs))
        torch_acts = torch.as_tensor(self.ep_as)
        torch_acts_one_hot = F.one_hot(torch_acts, num_classes=self.n_actions)
        torch_vt = torch.Tensor(discounted_ep_rs_norm).reshape(-1, 1)

        all_act_prob = self.policy_net(torch_obs)
        loss = torch.mean(-torch.log(all_act_prob) * torch_acts_one_hot * torch_vt)
        # criterion = F.cross_entropy(all_act_prob, torch_acts_one_hot)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.cost_his.append(loss.data.numpy())

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []  # empty episode data
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs
