import torch.nn as nn
# from common.config import NetworkConfig
# network_config = NetworkConfig()


class VectorNetwork(nn.Module):
    def __init__(self, input_dim=4, action_dim=2):
        super(VectorNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


if __name__ == "__main__":
    net = VectorNetwork(4, 2)

