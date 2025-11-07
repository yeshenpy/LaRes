import numpy as np
import random

# the replay memory
class replay_buffer:
    def __init__(self, memory_size):
        self.storge = []
        self.memory_size = memory_size
        self.next_idx = 0
    
    # add the samples
    def add(self, org_info, obs, action, reward_list, obs_, done):
        data = (org_info, obs, action, reward_list, obs_, done)
        if self.next_idx >= len(self.storge):
            self.storge.append(data)
        else:
            self.storge[self.next_idx] = data
        # get the next idx
        self.next_idx = (self.next_idx + 1) % self.memory_size
    
    # get the length of the replay buffer
    def get_length(self):
        return len(self.storge)

    # encode samples
    def _encode_sample(self, idx, index):
        obses, actions, rewards, obses_, dones = [], [], [], [], []
        for i in idx:
            data = self.storge[i]
            _, obs, action, reward_list,  obs_, done = data
            obses.append(np.array(obs, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward_list[index])
            obses_.append(np.array(obs_, copy=False))
            dones.append(done)
        return np.array(obses), np.array(actions), np.array(rewards), np.array(obses_), np.array(dones)
    
    # sample from the memory
    def sample(self, batch_size, index):

        idxes = [random.randint(0, len(self.storge) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes, index)

    def _encode_sample_elite(self, idx, index, elite_index, dynamic_weight):
        obses, actions, rewards, obses_, dones = [], [], [], [], []
        for i in idx:
            data = self.storge[i]
            _, obs, action, reward_list, obs_, done = data
            obses.append(np.array(obs, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward_list[index]* dynamic_weight  + (1- dynamic_weight)*reward_list[elite_index])
            obses_.append(np.array(obs_, copy=False))
            dones.append(done)
        return np.array(obses), np.array(actions), np.array(rewards), np.array(obses_), np.array(dones)

    def sample_with_elite(self, batch_size, index, elite_index, dynamic_weight):

        idxes = [random.randint(0, len(self.storge) - 1) for _ in range(batch_size)]
        return self._encode_sample_elite(idxes, index, elite_index, dynamic_weight)

    def sample_data_for_fau(self, batch_size, index):
        if len(self.storge) > batch_size:
            idxes = random.sample(list(range(len(self.storge))), batch_size)
        else:
            idxes = list(range(len(self.storge)))
        return self._encode_sample(idxes, index)