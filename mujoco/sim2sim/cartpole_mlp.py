import mujoco
import mujoco.viewer
import numpy as np
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ==== 神经网络模型定义 ====
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
        return action.clamp(-400.0, 400.0)

class FullModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a2c_network = A2CNetwork()

# ==== 加载模型 ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FullModel().to(device)
ckpt = torch.load("runs/CartpoleCmd.pth", map_location=device)
model.load_state_dict(ckpt["model"], strict=False)
model.eval()

# ==== 加载 MuJoCo 模型 ====
mjcf_path = "../assets/cartpole_cmd.xml"
model_mj = mujoco.MjModel.from_xml_path(mjcf_path)
data = mujoco.MjData(model_mj)

# ==== 滤波器 ====
from filter import LowPassFilter
Cart_Vel_LPF = LowPassFilter(0.995)

# ==== 传感器读取函数 ====
def get_sensor_value(name):
    sensor_id = mujoco.mj_name2id(model_mj, mujoco.mjtObj.mjOBJ_SENSOR, name)
    adr = model_mj.sensor_adr[sensor_id]
    dim = model_mj.sensor_dim[sensor_id]
    return data.sensordata[adr:adr + dim]

# ==== 数据记录 ====
angle_data = []
force_data = []
pos_data   = []
vel_data   = []
time_data  = []

# ==== 启动 Viewer ====
with mujoco.viewer.launch_passive(model_mj, data) as viewer:
    viewer.cam.distance = 10.0
    viewer.cam.azimuth = 0
    viewer.cam.elevation = -10
    viewer.cam.lookat[:] = [0, 0, 2]

    t = 0.0
    dt = 0.001
    while viewer.is_running():
        mujoco.mj_step(model_mj, data)

        start_time = time.time()  # 记录帧开始时间

        # 读取观测
        cart_pos   = get_sensor_value("cart_position")[0]
        cart_vel   = get_sensor_value("cart_velocity")[0]
        pole_angle = get_sensor_value("pole_angle")[0]
        pole_vel   = get_sensor_value("pole_velocity")[0]

        cart_vel_lpf = Cart_Vel_LPF.filter(cart_vel)

        command=0.0
        # 构造观测向量：cart_pos, cart_vel_lpf, pole_angle, pole_vel, command
        obs = np.array([cart_pos, cart_vel, pole_angle, pole_vel, command], dtype=np.float32)
        obs_tensor = torch.tensor(obs).unsqueeze(0).to(device)

        # 神经网络推理得到动作
        with torch.no_grad():
            action = model.a2c_network.act(obs_tensor)
        force = float(action.cpu().numpy()[0])

        # 设置控制量
        data.ctrl[0] = force*400

        # 记录数据
        pos_data.append(cart_pos)
        vel_data.append(cart_vel)
        angle_data.append(pole_angle)
        force_data.append(force)
        time_data.append(t)

        viewer.sync()
        time.sleep(dt)
        t += dt

        # === 计算和显示帧率 ===
        frame_time = time.time() - start_time
        fps = 1.0 / frame_time if frame_time > 0 else 0
        print(f"Time: {t:.2f} sec | FPS: {fps:.2f}")
        # viewer.title = f"CartPole - t={t:.2f}s | FPS={fps:.1f}"

# 仿真关闭后绘图
plt.figure(figsize=(10, 8))

plt.subplot(4, 1, 1)
plt.plot(time_data, pos_data, label="Cart Position (m)", color='green')
plt.ylabel("Cart Pos")
plt.xlabel("Time (s)")
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(time_data, vel_data, label="Cart Vel (m/s)")
plt.ylabel("Cart Vel")
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(time_data, angle_data, label="Pole Angle (rad)")
plt.ylabel("Angle")
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(time_data, force_data, label="Control Force (N)", color='orange')
plt.ylabel("Force")
plt.grid(True)
plt.legend()


plt.tight_layout()
plt.show()
