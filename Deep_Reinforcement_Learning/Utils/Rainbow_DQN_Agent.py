# Yongyang Liu

import random
import math
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

from Utils.Dueling_DQN_NeuralNetwork import DQN
from Utils.PrioritizedReplayBuffer import PER

class Agent(nn.Module):
    def __init__(self,
        input_shape,
        num_actions,
        device,
        PATH,
        gamma = 0.95,
        learning_rate = 0.001,
        replay_size = 10000,
        batch_size = 128
        ):
        super(Agent, self).__init__()

        self.device = device
        self.PATH = PATH
        self.gamma = gamma
        self.lr = learning_rate
        self.num_actions = num_actions

        epsilon_start = 1.0
        epsilon_final = 0.01
        epsilon_decay = 200
        self.epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

        self.replay_size = replay_size
        self.batch_size = batch_size

        self.policy_net = DQN(input_shape,num_actions).to(device)
        self.target_net = DQN(input_shape,num_actions).to(device)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

        self.replay_buffer = PER(replay_size)

        self.best_loss = 9999

    def declare_networks(self):
        self.policy_net = DQN(input_shape,num_actions).to(device)
        self.target_net = DQN(input_shape,num_actions).to(device)

    def declare_memory(self):
        self.replay_buffer = PER(self.replay_size)


    def compute_loss(self):
        if len(self.replay_buffer)>self.batch_size:
            state, action, reward, next_state, done, indices, weights = self.replay_buffer.sample(self.batch_size)

            state      = Variable(torch.Tensor(np.array(state))).to(self.device)
            action     = Variable(torch.LongTensor(action)).to(self.device)
            reward     = Variable(torch.Tensor(np.array(reward))).to(self.device)
            next_state = Variable(torch.Tensor(np.array(next_state))).to(self.device)
            done       = Variable(torch.Tensor(np.array(done))).to(self.device)
            weight    = Variable(torch.Tensor(np.array(weights))).to(self.device)

            q_values   = self.policy_net(state)
            q_value    = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                next_q_values = self.policy_net(next_state)
                next_q_state_values = self.target_net(next_state)
                next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)

            expected_q_value = reward + self.gamma * next_q_value * (1 - done)

            # MSE
            loss  = (q_value - expected_q_value.detach()).pow(2) * weight
            prios = loss + 1e-5
            loss  = loss.mean()

            self.optimizer.zero_grad()
            loss.backward()
            for param in self.policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
            self.replay_buffer.update_priorities(indices, prios.data.cpu().numpy())
            self.optimizer.step()

            if loss < self.best_loss:
                self.model_save()
                self.best_loss = loss

            return loss.item()
        else:
            return 9999

    def append_buffer(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)


    def get_action(self, state, episode):
        epsilon = self.epsilon_by_frame(episode)
        with torch.no_grad():
            if random.random() > epsilon:
                q_value = self.policy_net(state)
                action  = q_value.max(1)[1].item()
            else:
                action = np.random.randint(0, self.num_actions)

        return action

    def update_target_model(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def model_save(self):
        torch.save({
                    'model_state_dict': self.policy_net.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    }, self.PATH)

    def model_load(self):
        if self.device == "cuda:0":
            checkpoint = torch.load(self.PATH)
        else:
            checkpoint = torch.load(self.PATH, map_location = torch.device('cpu'))

        self.policy_net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
