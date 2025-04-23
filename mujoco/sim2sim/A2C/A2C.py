import torch
import torch.nn as nn
import numpy as np

OBS_DIM = 5
ACT_DIM = 1

class A2CNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor_mlp = nn.Sequential(
            nn.Linear(OBS_DIM, 32),
            nn.ELU(),
            nn.Linear(32, 32),
            nn.ELU()
        )
        self.mu = nn.Linear(32, ACT_DIM)
        self.value = nn.Linear(32, 1)
        self.sigma = nn.Parameter(torch.zeros(ACT_DIM), requires_grad=False)

    def forward(self, obs):
        x = self.actor_mlp(obs)
        mu = self.mu(x)
        value = self.value(x)
        std = self.sigma.exp()
        return mu, std, value

    def act(self, obs):
        mu, std, _ = self.forward(obs)
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        return action.clamp(-1.0, 1.0)

class FullModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a2c_network = A2CNetwork()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FullModel().to(device)

# 👇 加载 checkpoint 并提取模型参数
checkpoint = torch.load("CartpoleCmd.pth", map_location=device)
state_dict = checkpoint["model"]

# 加载状态字典
model.load_state_dict(state_dict, strict=False)
model.eval()

# 示例观测值
obs = np.array([0.0, 0.1, 0.2, 0.3, 0.0], dtype=np.float32)
obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
with torch.no_grad():
    action = model.a2c_network.act(obs_tensor)
print("Predicted Action:", action.cpu().numpy()[0])
