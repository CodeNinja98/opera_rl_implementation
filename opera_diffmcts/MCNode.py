import torch as th
class MCNode:
    NUM_ACTS = 4
    def __init__(self,S,rewards=None):
        self.S = S
        self.branches= [None]*self.NUM_ACTS
        self.rewards = th.zeros(self.NUM_ACTS) if rewards is None else rewards
        self.sa_rewards = [None for _ in range(self.NUM_ACTS)]
    def V(self):
        return max([v for v in self.rewards if v!=0])