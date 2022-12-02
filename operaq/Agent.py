import random
import numpy as np
from collections import deque
from QNet import QNet
import torch as th
from numpy.random import default_rng
rng = default_rng()
import random
import copy

ACT_SPACE = 4
OBS_SPACE = 8
BATCH_SIZE = 128
MEM_LEN = 15000
SAMPLE_SIZE = 64
WARMUP =5500
s0_REPLACE = 0.2
class Agent():
    def __init__(self,H):
        self.memory = deque(maxlen=MEM_LEN)
        self.last_state = None
        self.qnet = QNet().cuda()
        #PLACEHOLDER
        self.policyQ = None
        self.H = H
        self.last_act = None
        self.s0 = None

    def step(self,obs,reward):
        '''
        if len(self.episodeMemory) <= 3:
            act = random.randint(0,ACT_SPACE-1)
        '''
        if self.s0 is None or rng.random() <= s0_REPLACE and len(self.curr_ep) == 0:
            self.s0 = th.Tensor(obs).cuda()
        if self.policyQ is not None:
            with th.no_grad(): 
                self.policyQ.eval()
                qvals = self.getQVals(th.Tensor(obs).cuda(),self.policyQ)
                '''
                if len(self.memory) > WARMUP:
                    act = np.argmax(qvals).item()
                else:
                    action_probs = th.nn.functional.softmax(th.nan_to_num(qvals),dim=0).numpy()
                    #print(action_probs)
                    act = rng.choice(ACT_SPACE,p=action_probs)
                '''
                act = np.argmax(qvals).item()

        else:
            act = rng.choice(ACT_SPACE)
        
        if self.last_state is None:
            self.last_state = obs
            self.last_act = act
        else:
            transition = (self.last_state,self.last_act,obs,reward,False)
            self.curr_ep.append(transition)
            self.last_state = obs
            self.last_act = act
        return act
    
    def getQVals(self, S, Q):
        return th.Tensor([Q(th.cat([S if k == A else th.zeros(OBS_SPACE).cuda() for k in range(ACT_SPACE)]) ) for A in range(ACT_SPACE) ])


    def endEpisode(self,obs,reward):
        transition = (self.last_state,self.last_act,obs,reward,True)
        self.curr_ep.append(transition)
        #if len(self.episodeMemory) < 3: return
        sample = random.sample(self.curr_ep,SAMPLE_SIZE if len(self.curr_ep) > SAMPLE_SIZE else len(self.curr_ep) )
        #sample = self.curr_ep
        for v in sample: self.memory.append(v)

        last_ep = list(self.memory)
        X = np.zeros((len(last_ep),OBS_SPACE*ACT_SPACE))
        X_prime = np.zeros((ACT_SPACE,len(last_ep),OBS_SPACE*ACT_SPACE))
        r = np.zeros(len(last_ep))

        terminal_mask = th.ones((len(last_ep),1)).cuda()

        for i in range(len(last_ep)):
            X[i] = np.concatenate([last_ep[i][0] if k == last_ep[i][1] else np.zeros(OBS_SPACE) for k in range(ACT_SPACE)])
            r[i] = last_ep[i][3]
            if last_ep[i][4]: terminal_mask[i] = 0
            for A in range(ACT_SPACE):
                X_prime[A][i] =np.concatenate([last_ep[i][2] if k == A else np.zeros(OBS_SPACE) for k in range(ACT_SPACE)])
        #print(terminal_mask)
        X = th.Tensor(X).cuda()
        r = th.Tensor(r).cuda()
        X_prime = th.Tensor(X_prime).cuda()

        self.trainQNet(X,r.unsqueeze(1),X_prime,terminal_mask)
        self.policyQ = self.makeOptimismNet(X)

        print("Base V(s0):",th.max(th.stack([self.qnet(th.cat([self.s0 if k == A else th.zeros(OBS_SPACE).cuda() for k in range(ACT_SPACE)])) for A in range(ACT_SPACE) ])),
        "Optimistic V(s0):",
        th.max(th.stack([self.policyQ(th.cat([self.s0 if k == A else th.zeros(OBS_SPACE).cuda() for k in range(ACT_SPACE)])) for A in range(ACT_SPACE) ]))
        )

    def makeOptimismNet(self,X):

        criterion = th.nn.HuberLoss().cuda()
        optimismNet = QNet().cuda()
        optimismNet.load_state_dict(self.qnet.state_dict())
        optimismNet.fc4.weight.requires_grad=False
        optimismNet.fc4.bias.requires_grad=False

        opt = th.optim.SGD(optimismNet.parameters(),lr=0.001,maximize=True)
        opt.zero_grad()
        s0 = th.Tensor(self.s0).detach()
        qval = th.max(th.stack([optimismNet(th.cat([s0 if k == A else th.zeros(OBS_SPACE).cuda() for k in range(ACT_SPACE)])) for A in range(ACT_SPACE) ]))
        print(qval)
        #loss = criterion(qval,th.Tensor([200]))
        #loss.backward()
        qval.backward()
        opt.step()
        return optimismNet

    def beginEpisode(self):
        self.curr_ep = deque()


    def trainQNet(self,X,r,Xp,terminal_mask,epochs = 5):

        opt = th.optim.Adam(self.qnet.parameters(),lr=0.01)
        criterion = th.nn.HuberLoss().cuda()

        self.qnet.train()
        zeros = th.zeros((len(X),1)).cuda()

        for epoch in range(epochs):
            opt.zero_grad()
            #print(self.qnet(X).shape, r.shape,th.amax(th.stack([self.qnet(Xp[i]) for i in range(ACT_SPACE)]),0).shape )
            #print(th.stack([self.qnet(Xp[i]) for i in range(ACT_SPACE)]))
            bell_error = self.qnet(X) - r - terminal_mask * th.amax(th.stack([self.qnet(Xp[i]) for i in range(ACT_SPACE)]),0)
            loss = criterion(bell_error,zeros)
            if epoch % 500 == 0: print('Q Loss:',loss.item())
            loss.backward()
            opt.step()

            





