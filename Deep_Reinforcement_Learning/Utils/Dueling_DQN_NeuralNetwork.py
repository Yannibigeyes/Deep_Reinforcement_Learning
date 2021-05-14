#Yongyang Liu
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions

        self.conv1 = nn.Conv2d(1, 32, kernel_size=2, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2, stride=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=2, stride=1)

        
        self.h   = self.feature_size(self.feature_size(self.feature_size(input_shape[0], 2), 2), 2)
        self.w   = self.feature_size(self.feature_size(self.feature_size(input_shape[1], 2), 2), 2)
        #wide = 5-1-1-1 = 2
        
        self.val1 = nn.Linear(64*self.h*self.w, 512)
        self.val2 = nn.Linear(512, self.num_actions)
        
        self.adv1 = nn.Linear(64*self.h*self.w, 512)
        self.adv2 = nn.Linear(512, 1)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        
        
        value = F.relu(self.val1(x))
        value = self.val2(value)
        
        advantage = F.relu(self.adv1(x))
        advantage = self.adv2(advantage)
    
        return value + advantage  - advantage.mean()
    
    def feature_size(self, size, kernel_size, stride = 1, padding = 0):
        return int((size - kernel_size + 2*padding)/ stride  + 1)
