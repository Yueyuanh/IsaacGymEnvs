import mujoco
import mujoco.viewer
import numpy as np
import time

mjcf_path = "../assets/cartpole_cmd.xml"
model = mujoco.MjModel.from_xml_path(mjcf_path)
data = mujoco.MjData(model)

# PID参数
kp, ki, kd = 1000, 1, 10
target_angle = 0.0  # 竖直
integral = 0
last_error = 0

def get_sensor_value(name):
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    adr = model.sensor_adr[sensor_id]
    dim = model.sensor_dim[sensor_id]
    return data.sensordata[adr:adr + dim]

# 启动 viewer
with mujoco.viewer.launch_passive(model, data) as viewer:

    # 设置相机参数
    viewer.cam.distance = 10.0      # 相机距离
    viewer.cam.azimuth = 0        # 水平旋转角（0-360）
    viewer.cam.elevation = -10     # 俯仰角（上下视角）
    viewer.cam.lookat[:] = [0, 0, 2]  # 注视点坐标


    while 1:
        mujoco.mj_step(model, data)

        # 读取角度
        pole_angle = get_sensor_value("pole_angle")[0]
        error = target_angle - pole_angle
        integral += error * model.opt.timestep
        derivative = (error - last_error) / model.opt.timestep
        last_error = error

        # PID输出
        force = kp * error + ki * integral + kd * derivative
        force = np.clip(force, -10, 10)

        data.ctrl[0] = force

        # 每步更新viewer
        viewer.sync()

        time.sleep(model.opt.timestep)
