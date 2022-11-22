import torch as th

class DynamicsNet(th.nn.Module):
    def __init__(self):
        super(DynamicsNet, self).__init__()
        self.fc1 = th.nn.Linear(12, 12)
        self.fc2 = th.nn.Linear(12, 12)
        self.fc3 = th.nn.Linear(12, 10)
        self.fc4 = th.nn.Linear(10, 9)
        self.fc5 = th.nn.Linear(9,8)

    def forward(self, x):
        x = th.relu(self.fc1(x))
        x = th.relu(self.fc2(x))
        x = th.relu(self.fc3(x))
        x = th.relu(self.fc4(x))
        x = self.fc5(x)
        return x