import random
import numpy as np
from collections import deque
from DynamicsNet import DynamicsNet
import torch as th

ACT_SPACE = 4
OBS_SPACE = 8
BATCH_SIZE = 32
class Agent():
    def __init__(self,H):
        self.episodeMemory = deque()
        self.last_state = None
        self.dnet = DynamicsNet()
        self.H = H
    def step(self,obs,reward):
        act = random.randint(0,3)
        if self.last_state is None:
            self.last_state = obs
        else:
            transition = (self.last_state,act,obs,reward)
            self.episodeMemory[-1].append(transition)
            self.last_state = obs
        return act

    def endEpisode(self):
        last_ep = list(self.episodeMemory[-1])
        X = np.zeros((len(last_ep),OBS_SPACE+ACT_SPACE))
        y = np.zeros((len(last_ep),OBS_SPACE))
        for i in range(len(last_ep)):
            act_onehot = np.zeros(ACT_SPACE)
            act_onehot[last_ep[i][1]] = 1
            X[i] = np.concatenate((last_ep[i][0],act_onehot))
            y[i] = last_ep[i][2]
        self.makeDynamicsNet(X,y)
    def beginEpisode(self):
        self.episodeMemory.append(deque())

    def makeDynamicsNet(self,X,y,epochs = 100):
        dataloader = th.utils.data.DataLoader(th.utils.data.TensorDataset(th.Tensor(X).type(th.FloatTensor),
        th.Tensor(y).type(th.FloatTensor)), batch_size=BATCH_SIZE,shuffle=True)

        model = DynamicsNet()
        opt = th.optim.Adam(model.parameters(),lr=0.001)
        criterion = th.nn.MSELoss()

        for epoch in range(epochs):
            for data in dataloader:
                X, y = data
                opt.zero_grad()
                output = model(X)
                loss = criterion(output,y)
                print('Loss:',loss.item())
                loss.backward()
                opt.step()
        return model




