import torch as th

class DynamicsNet(th.nn.Module):
    def __init__(self):
        super(DynamicsNet, self).__init__()
        self.fc1 = th.nn.Linear(32, 32)
        self.fc2 = th.nn.Linear(32, 32)
        #self.fc3 = th.nn.Linear(32, 32)
        self.fc6 = th.nn.Linear(32, 32)
        self.fc5 = th.nn.Linear(32,9)

    def forward(self, x):
        x = th.relu(self.fc1(x)) 
        x = th.relu(self.fc2(x)) 
        #z = th.relu(self.fc3(z)) 
        x = th.relu(self.fc6(x))
        x = self.fc5(x)
        return x