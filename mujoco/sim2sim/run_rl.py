import mujoco
import mujoco.viewer
import numpy as np
import torch
import time

from mlp import A2CNetwork  # 模型定义与你前面一致

# 加载训练模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = A2CNetwork().to(device)
state_dict = torch.load("runs/CartpoleCmd.pth", map_location=device)
model.load_state_dict(state_dict["model"])  # 只加载模型部分
model.eval()

# 初始化 MuJoCo 模型
mjcf_path = "../assets/cartpole_cmd.xml"
model_mj = mujoco.MjModel.from_xml_path(mjcf_path)
data = mujoco.MjData(model_mj)

model_mj.opt.timestep = 0.001
sim_time = 0.0
refresh_interval = 1.0 / 60  # 每 1/60 秒刷新一次 Viewer
last_refresh_time = 0.0

# 获取观测
def get_obs():
    cart_pos = dA2CNetworkta.sensordata[3]
    return np.array([cart_pos, cart_vel, pole_ang, pole_vel, 0], dtype=np.float32)

# 帧率计算
frame_counter = 0
fps_start_time = time.time()

# 启动 Viewer
with mujoco.viewer.launch_passive(model_mj, data) as viewer:
    viewer.cam.distance = 10.0
    viewer.cam.azimuth = 0
    viewer.cam.elevation = -10
    viewer.cam.lookat[:] = [0, 0, 2]

    while viewer.is_running():
        # 获取状态输入，推理动作
        obs = get_obs()
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action = model(obs_tensor)[0].cpu().numpy()[0]

        # 应用动作
        data.ctrl[0] = action

        # 物理模拟前进一步
        mujoco.mj_step(model_mj, data)
        sim_time += model_mj.opt.timestep

        # 每隔一段模拟时间刷新 Viewer
        if sim_time - last_refresh_time >= refresh_interval:
            viewer.sync()
            last_refresh_time = sim_time

        # 计算并打印 FPS
        frame_counter += 1
        elapsed = time.time() - fps_start_time
        if elapsed >= 1.0:
            print(f"Sim Time: {sim_time:.2f} sec | FPS: {frame_counter / elapsed:.2f}")
            frame_counter = 0
            fps_start_time = time.time()
