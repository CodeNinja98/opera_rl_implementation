import random
import numpy as np
from collections import deque
from DynamicsNet import DynamicsNet
import torch as th
from scipy.special import softmax
from MCNode import MCNode
from numpy.random import default_rng
rng = default_rng()


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
        act = random.randint(0,ACT_SPACE-1)
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

    def is_terminal(self,S):
        #PLACEHOLDER
        return False

    def predict_state(self,S,A,dynamics):
        #PLACEHOLDER
        return rng.random(OBS_SPACE)
    
    def differentiableMCQ(self,initial_state,dynamics,reward,N=500,max_depth=100,warmup=150):
        initial_node = MCNode(initial_state)
        for n in range(N):
            curr_node = initial_node
            curr_rollout = deque()
            curr_rollout.append((initial_node,-1))

            while True:
                if n<warmup: action = random.randint(0,ACT_SPACE-1)
                else: 
                    action_probs = softmax(curr_node.rewards)
                    action = rng.choice(ACT_SPACE,p=action_probs)
                
                #Transition to next state
                if curr_node.branches[action] is not None:
                    curr_node = curr_node.branches[action]
                else:
                    predicted_state = self.predict_state(curr_node.S,action,dynamics)
                    curr_node = MCNode(predicted_state)
                
                curr_rollout.append((curr_node,action))
                if len(curr_rollout) >= max_depth or self.is_terminal(curr_node.S):
                    break

            #Backprop to update state-action values


            





