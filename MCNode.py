import torch as th
class MCNode:
    NUM_ACTS = 4
    def __init__(self,S,R,rewards=None):
        self.S = S
        self.branches= [None]*self.NUM_ACTS
        self.rewards = th.zeros(self.NUM_ACTS) if rewards is None else rewards
        self.sa_rewards = th.Tensor([R(S,a) for a in range(self.NUM_ACTS)])
    def V(self):
        return th.max(self.rewards)