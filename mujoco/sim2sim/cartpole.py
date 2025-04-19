import mujoco
import mujoco.viewer
import numpy as np
import pygame

# 读取你的 MJCF 文件路径
mjcf_path = "../assets/cartpole_cmd.xml"

# 加载 MuJoCo 模型
model = mujoco.MjModel.from_xml_path(mjcf_path)
data = mujoco.MjData(model)

# 初始化 MuJoCo 可视化窗口
with mujoco.viewer.launch_passive(model, data) as viewer:

    # 设置相机参数
    viewer.cam.distance = 10.0      # 相机距离
    viewer.cam.azimuth = 0        # 水平旋转角（0-360）
    viewer.cam.elevation = -10     # 俯仰角（上下视角）
    viewer.cam.lookat[:] = [0, 0, 2]  # 注视点坐标

    # 初始化 pygame 监听键盘
    pygame.init()
    screen = pygame.display.set_mode((400, 300))
    pygame.display.set_caption("CartPole Control")


    running = True
    force = 0.0  # 施加的控制力

    while running:
        # 监听键盘事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    force = -20  # 向左施加力
                elif event.key == pygame.K_RIGHT:
                    force = 20  # 向右施加力
            elif event.type == pygame.KEYUP:
                if event.key in (pygame.K_LEFT, pygame.K_RIGHT):
                    force = 0.0  # 松开键，力归零

        # 施加外力到 `cart` (假设 cart 关节是 `slider_to_cart`)
        data.ctrl[0] = force
        print(data.ctrl[0])

        # 运行 MuJoCo 仿真
        # mujoco.mj_step(model, data)

        # 更新 MuJoCo 视图
        viewer.sync()
    
    pygame.quit()
