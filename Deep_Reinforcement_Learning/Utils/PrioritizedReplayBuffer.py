#Yongyang Liu
import random
import numpy as np
class PER(object):
    def __init__(self, capacity, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.capacity   = capacity
        self.buffer     = []
        self.pos        = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, state, action, reward, state_next, done):

        state      = np.expand_dims(state, 0)
        state_next = np.expand_dims(state_next, 0)
        
        max_prio = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, state_next, done))
        else:
            self.buffer[self.pos] = (state, action, reward, state_next, done)
        
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
        
        if len(self.buffer) > self.capacity:
            del self.buffer[0]
    
    def sample(self, batch_size, beta=0.4):
    
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        
        probs  = prios ** self.prob_alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        total    = len(self.buffer)
        weights  = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights  = np.array(weights, dtype=np.float32)
        
        state, action, reward, state_next, done = zip(*samples)
        return np.concatenate(state), action, reward, np.concatenate(state_next), done, indices, weights
        
    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio 
    
    def delete(self):
        self.buffer.clear()
    
    def __len__(self):
        return len(self.buffer)