import torch as th

class RewardNet(th.nn.Module):
    def __init__(self):
        super(RewardNet, self).__init__()
        self.fc1 = th.nn.Linear(12, 12)
        self.fc2 = th.nn.Linear(12, 12)
        self.fc3 = th.nn.Linear(12, 10)
        self.fc4 = th.nn.Linear(10, 5)
        self.fc5 = th.nn.Linear(5,1)

    def forward(self, x):
        x = th.relu(self.fc1(x))
        x = th.relu(self.fc2(x))
        x = th.relu(self.fc3(x))
        x = th.relu(self.fc4(x))
        x = self.fc5(x)
        return x