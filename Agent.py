import random
import numpy as np
from collections import deque
from DynamicsNet import DynamicsNet
import torch as th
from MCNode import MCNode
from numpy.random import default_rng
from RewardNet import RewardNet


ACT_SPACE = 4
OBS_SPACE = 8
BATCH_SIZE = 128
TERM_BOUND = -5
class Agent():
    def __init__(self,H):
        self.episodeMemory = deque()
        self.last_state = None
        self.dnet = DynamicsNet()
        self.rnet = RewardNet()
        self.H = H
        self.last_act = None

    def step(self,obs,reward):
        if len(self.episodeMemory) <= 3:
            act = random.randint(0,ACT_SPACE-1)
        else: 
            with th.no_grad(): 
                self.dnet.eval()
                self.rnet.eval()
                R = lambda s,a : self.rnet(s)[a]
                qvals = self.differentiableMCQ(th.Tensor(obs),self.dnet,R)
                act = np.argmax(qvals).item()
        
        if self.last_state is None:
            self.last_state = obs
            self.last_act = act
        else:
            transition = (self.last_state,self.last_act,obs,reward)
            self.episodeMemory[-1].append(transition)
            self.last_state = obs
            self.last_act = act
        return act

    def endEpisode(self,obs,reward):
        transition = (self.last_state,self.last_act,np.array([-5]*OBS_SPACE),reward)
        self.episodeMemory[-1].append(transition)
        if len(self.episodeMemory) < 3: return

        last_ep = list(self.episodeMemory[-1]) + list(self.episodeMemory[-2]) + list(self.episodeMemory[-3])
        X = np.zeros((len(last_ep),OBS_SPACE+ACT_SPACE))
        y = np.zeros((len(last_ep),OBS_SPACE))

        X_reward = np.zeros((len(last_ep),OBS_SPACE))
        y_reward = np.zeros(len(last_ep))

        actions = np.zeros(len(last_ep))

        for i in range(len(last_ep)):
            act_onehot = np.zeros(ACT_SPACE)
            act_onehot[last_ep[i][1]] = 1
            X[i] = np.concatenate((last_ep[i][0],act_onehot))
            y[i] = last_ep[i][2]
            X_reward[i] = last_ep[i][0]
            actions[i] = last_ep[i][1]

            y_reward[i] = last_ep[i][3]
        X = th.Tensor(X)
        self.trainDynamicsNet(X,th.Tensor(y))
        self.trainRewardNet(th.Tensor(X_reward),th.Tensor(y_reward).unsqueeze(1),th.Tensor(actions).type(th.LongTensor).unsqueeze(1)) 

    def beginEpisode(self):
        self.episodeMemory.append(deque())

    def trainRewardNet(self, X, y, actions, epochs = 5000):

        opt = th.optim.Adam(self.rnet.parameters(),lr=0.001)
        criterion = th.nn.MSELoss()

        self.rnet.train()

        for epoch in range(epochs):
            opt.zero_grad()
            output = self.rnet(X).gather(1,actions)
            loss = criterion(output,y)
            #print('Reward Loss:',loss.item())
            loss.backward()
            opt.step()
        print(y)
        print(self.rnet(X).gather(1,actions))
        print("Final Reward Error:",th.abs((y - self.rnet(X).gather(1,actions))).mean(axis=0) )

    def trainDynamicsNet(self,X,y,epochs = 1000):

        opt = th.optim.Adam(self.dnet.parameters(),lr=0.01)
        criterion = th.nn.MSELoss()

        self.dnet.train()

        for epoch in range(epochs):
            opt.zero_grad()
            output = self.dnet(X)
            loss = criterion(output,y)
            if epoch % 100 == 0: print('Dynamics Loss:',loss.item())
            loss.backward()
            opt.step()

    def is_terminal(self,S):
        return any(S<TERM_BOUND)

    def predict_state(self,S,A,dynamics):
        act_onehot = th.zeros(ACT_SPACE)
        act_onehot[A] = 1
        return dynamics(th.cat((S,act_onehot)) )
    
    def differentiableMCQ(self,initial_state,dynamics,reward,N=75,max_depth=50,warmup=50):
        initial_node = MCNode(initial_state, reward)
        for n in range(N):
            curr_node = initial_node
            curr_rollout = deque()
            curr_rollout.append((initial_node,-1))

            while True:
                if n<warmup: action = th.randint(ACT_SPACE,(1,))
                else: 
                    action_probs = th.nn.functional.softmax(th.nan_to_num(curr_node.rewards),dim=0)
                    action = th.multinomial(action_probs,1)
                
                #Transition to next state
                if curr_node.branches[action] is not None:
                    curr_node = curr_node.branches[action]
                else:
                    predicted_state = self.predict_state(curr_node.S,action,dynamics)
                    curr_node = MCNode(predicted_state,reward)
                
                curr_rollout.append((curr_node,action))
                if len(curr_rollout) >= max_depth or self.is_terminal(curr_node.S):
                    break

            #Backprop to update state-action values
            #Fill in values for last node
            final_act = curr_rollout.pop()[1]
            curr_rollout[-1][0].rewards[final_act] = curr_rollout[-1][0].sa_rewards[final_act]

            while len(curr_rollout) > 1:
                node, act = curr_rollout.pop()
                curr_rollout[-1][0].rewards[act] = curr_rollout[-1][0].sa_rewards[act] + node.V()
        return initial_node.rewards



            





