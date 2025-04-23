import mujoco
import mujoco.viewer
import numpy as np
import time
import matplotlib.pyplot as plt

from pid import PIDController
from filter import LowPassFilter

# 模型加载
mjcf_path = "../assets/cartpole_cmd.xml"
model = mujoco.MjModel.from_xml_path(mjcf_path)
data = mujoco.MjData(model)

# PID参数
Pos_PID_Param=[1,0,2]
Spd_PID_Param=[0.12,0,0.01]
Ang_PID_Param=[6500,0,100]
dt=0.001



# PID实例
Position_PID = PIDController(*Pos_PID_Param,derivative_filter_alpha=0.999,output_limit=3)
Speed_PID    = PIDController(*Spd_PID_Param,derivative_filter_alpha=0.999,output_limit=0.5)
Angle_PID    = PIDController(*Ang_PID_Param,output_limit=10000)

Cart_Vel_LPF =LowPassFilter(0.995)
# 传感器读取函数
def get_sensor_value(name):
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    adr = model.sensor_adr[sensor_id]
    dim = model.sensor_dim[sensor_id]
    return data.sensordata[adr:adr + dim]

# 数据记录
angle_data = []
force_data = []
pos_data   = []
vel_data   = []
time_data  = []

# 启动 MuJoCo Viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 10.0
    viewer.cam.azimuth = 0
    viewer.cam.elevation = -10
    viewer.cam.lookat[:] = [0, 0, 2]

    t = 0.0
    while viewer.is_running():
        mujoco.mj_step(model, data)
        start_time = time.time()  # 记录帧开始时间


        cart_pos   = get_sensor_value("cart_position")[0]
        cart_vel   = get_sensor_value("cart_velocity")[0]
        pole_angle = get_sensor_value("pole_angle")[0]
        pole_vel   = get_sensor_value("pole_velocity")[0]

        cart_vel_lpf = Cart_Vel_LPF.filter(cart_vel)

        if t>=1:
            pos_set = 1
        else:
            pos_set = 0
        # pos_set=0
        speed_set = Position_PID.compute(pos_set,-cart_pos) 
        
        angle_set = Speed_PID.compute(speed_set,-cart_vel_lpf)
        # print(angle_set,"  ",cart_vel_lpf)
        # angle_set = 0.1
        force = Angle_PID.compute(angle_set, pole_angle)
        data.ctrl[0] = force
        # print(angle_set,"  ",cart_vel)

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
