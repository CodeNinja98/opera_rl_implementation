import random
import numpy as np
from collections import deque
from DynamicsNet import DynamicsNet
import torch as th
from MCNode import MCNode
from numpy.random import default_rng
rng = default_rng()
import random

ACT_SPACE = 4
OBS_SPACE = 8
BATCH_SIZE = 128
MEM_LEN = 15000
SAMPLE_SIZE = 100
class Agent():
    def __init__(self,H):
        self.memory = deque(maxlen=MEM_LEN)
        self.last_state = None
        self.dnet = DynamicsNet()
        self.H = H
        self.last_act = None

    def step(self,obs,reward):
        '''
        if len(self.episodeMemory) <= 3:
            act = random.randint(0,ACT_SPACE-1)
        '''
        with th.no_grad(): 
            self.dnet.eval()
            qvals = self.differentiableMCQ(th.Tensor(obs),self.dnet)
            act = np.argmax(qvals).item()
                #action_probs = th.nn.functional.softmax(th.nan_to_num(qvals),dim=0).numpy()
                #print(action_probs)
                #act = rng.choice(4,p=action_probs)
        
        if self.last_state is None:
            self.last_state = obs
            self.last_act = act
        else:
            transition = (self.last_state,self.last_act,obs,reward,False)
            self.curr_ep.append(transition)
            self.last_state = obs
            self.last_act = act
        return act

    def endEpisode(self,obs,reward):
        transition = (self.last_state,self.last_act,obs,reward,True)
        self.curr_ep.append(transition)
        #if len(self.episodeMemory) < 3: return
        sample = random.sample(self.curr_ep,SAMPLE_SIZE if len(self.curr_ep) > SAMPLE_SIZE else len(self.curr_ep) )
        for v in sample: self.memory.append(v)

        last_ep = list(self.memory)
        X = np.zeros((len(last_ep),OBS_SPACE*ACT_SPACE))
        y = np.zeros((len(last_ep),OBS_SPACE+2))

        actions = np.zeros(len(last_ep))

        for i in range(len(last_ep)):
            X[i] = np.concatenate([last_ep[i][0] if k == last_ep[i][1] else np.zeros(OBS_SPACE) for k in range(ACT_SPACE)])
            y[i] = np.concatenate((last_ep[i][2],np.array([last_ep[i][3]]), np.array(([1])) if last_ep[i][4] else np.array(([0])) ) )
            
            actions[i] = last_ep[i][1]

        X = th.Tensor(X)
        self.trainDynamicsNet(X,th.Tensor(y))

    def beginEpisode(self):
        self.curr_ep = deque()


    def trainDynamicsNet(self,X,y,epochs = 8000):

        opt = th.optim.Adam(self.dnet.parameters(),lr=0.0001)
        criterion = th.nn.MSELoss()

        self.dnet.train()

        for epoch in range(epochs):
            opt.zero_grad()
            output = self.dnet(X)
            loss = criterion(output,y)
            if epoch % 500 == 0: print('Dynamics Loss:',loss.item())
            loss.backward()
            opt.step()

    def predict_state(self,S,A,dynamics):
        #act_onehot = th.zeros(ACT_SPACE)
        #act_onehot[A] = 1
        pred = dynamics(th.cat([S[:OBS_SPACE] if k == A else th.zeros(OBS_SPACE) for k in range(ACT_SPACE)]) )
        #state, reward, terminal
        return pred[:OBS_SPACE], pred[-2], pred[-1] > 0.5
    
    def differentiableMCQ(self,initial_state,dynamics,N=40,max_depth=30,warmup=10000):
        initial_node = MCNode(initial_state)
        for n in range(N):
            curr_node = initial_node
            curr_rollout = deque()
            curr_rollout.append((initial_node,-1))
            #print("N:",n,initial_node.rewards)

            while True:
                if n<warmup: action = th.randint(ACT_SPACE,(1,))
                else: 
                    action_probs = th.nn.functional.softmax(th.nan_to_num(curr_node.rewards),dim=0)
                    action = th.multinomial(action_probs,1)
                
                #Transition to next state
                if curr_node.branches[action] is not None:
                    curr_node = curr_node.branches[action]
                else:
                    predicted_state, r, t = self.predict_state(curr_node.S,action,dynamics)
                    curr_node = MCNode(predicted_state)
                    curr_rollout[-1][0].branches[action] = curr_node
                    curr_rollout[-1][0].sa_rewards[action] = r
                    #print("add node")
                
                curr_rollout.append((curr_node,action))
                if len(curr_rollout) >= max_depth or t:
                    break

            #Backprop to update state-action values
            #Fill in values for last node
            final_act = curr_rollout.pop()[1]
            curr_rollout[-1][0].rewards[final_act] = curr_rollout[-1][0].sa_rewards[final_act]

            while len(curr_rollout) > 1:
                node, act = curr_rollout.pop()
                curr_rollout[-1][0].rewards[act] = curr_rollout[-1][0].sa_rewards[act] + node.V()
        return initial_node.rewards



            





