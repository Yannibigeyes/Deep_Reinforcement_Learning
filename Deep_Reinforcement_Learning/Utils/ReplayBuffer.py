#Yongyang Liu
import random
import numpy as np

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer   = []
        self.capacity = capacity

    def push(self, state, action, reward, state_next, done):
        state      = np.expand_dims(state, 0)
        state_next = np.expand_dims(state_next, 0)
            
        self.buffer.append((state, action, reward, state_next, done))
        if len(self.buffer) > self.capacity:
            del self.buffer[0]
    
    def sample(self, batch_size):
        state, action, reward, state_next, done = zip(*random.sample(self.buffer, batch_size))

        return np.concatenate(state), action, reward, np.concatenate(state_next), done
    
    def delete(self):
        self.buffer.clear()
    
    def __len__(self):
        return len(self.buffer)
       
