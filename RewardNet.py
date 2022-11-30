import torch as th

class RewardNet(th.nn.Module):
    def __init__(self):
        super(RewardNet, self).__init__()
        self.fc1 = th.nn.Linear(8,64)
        self.fc2 = th.nn.Linear(64, 64)
        #self.fc3 = th.nn.Linear(12, 12)
        #self.fc4 = th.nn.Linear(12, 12)
        #self.fc5 = th.nn.Linear(12, 12)
        self.fc6 = th.nn.Linear(64,4)



    def forward(self, x):
        x = th.relu(self.fc1(x))
        x = th.relu(self.fc2(x))
        #x = th.relu(self.fc3(x))
        #x = th.relu(self.fc4(x))
        #x = th.relu(self.fc5(x))


        x = self.fc6(x)
        return x