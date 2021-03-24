import torch

class Model(torch.nn.Module):
    def __init__ (self):
        super().__init__()
        self.convlayers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.Flatten()
        )
        self.linearlayers = torch.nn.Sequential(
            torch.nn.Linear(1016064,1)
        )
    def forward(self, X):
        X = self.convlayers(X)
        X = self.linearlayers(X)
        return X
