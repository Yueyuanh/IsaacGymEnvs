import isaacgym
import isaacgym.gymapi as gymapi
import isaacgym.gymutil as gymutil
from isaacgym import gymtorch
import torch
import numpy as np
import os

# 计算 URDF 资源路径
asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets/urdf/")
asset_file = "cartpole_cmd.urdf"
# asset_file = "yummy/urdf/Yummy_Robot.urdf"

class CartPoleEnv:
    def __init__(self, urdf_path, urdf_file):
        """初始化 Isaac Gym 环境"""
        self.gym = gymapi.acquire_gym()
        self.urdf_path = urdf_path
        self.urdf_file = urdf_file
        self.sim = None
        self.env = None
        self.actor_handle = None
        self.viewer = None  # GUI 窗口

        # 物理参数
        self.sim_params = gymapi.SimParams()
        self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        self.sim_params.physx.solver_type = 1
        self.sim_params.physx.num_position_iterations = 8
        self.sim_params.physx.num_velocity_iterations = 1
        self.sim_params.use_gpu_pipeline = False
        self.sim_params.up_axis = gymapi.UP_AXIS_Z



        self.init_simulation()

    def init_simulation(self):
        """创建仿真环境并载入模型"""
        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, self.sim_params)
        if self.sim is None:
            raise Exception("Failed to create simulation")

       # 创建 Viewer（可视化窗口）
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if self.viewer is None:
            raise Exception("Failed to create viewer")

        # 添加地面
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)

        # 载入 URDF 资源，固定 base_link
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True  # 固定 base_link
        asset_options.armature = 0.01
        cartpole_asset = self.gym.load_asset(self.sim, self.urdf_path, self.urdf_file, asset_options)
        self.num_dof=self.gym.get_asset_dof_count(cartpole_asset)
        print("*******")
        print(self.num_dof)

        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]


        # 创建环境
        self.env = self.gym.create_env(self.sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 2), 1)

        # 设定初始位置
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, 2)
        pose.r = gymapi.Quat(0, 0, 0, 1)

        self.actor_handle = self.gym.create_actor(self.env, cartpole_asset, pose, "cartpole", 0, 1)

        # 观测器
        dof_props = self.gym.get_actor_dof_properties(self.env, self.actor_handle)
        dof_props['driveMode'][0] = gymapi.DOF_MODE_EFFORT #扭矩模式
        dof_props['driveMode'][1] = gymapi.DOF_MODE_NONE
        dof_props['stiffness'][:] = 0.0
        dof_props['damping'][:] = 0.0
        self.gym.set_actor_dof_properties(self.env, self.actor_handle, dof_props)


        # 设置摄像机位置
        cam_pos = gymapi.Vec3(5.0, 0.0, 3.0)  # 摄像机放置在 (5, 0, 3)
        cam_target = gymapi.Vec3(0.0, 0.0, 2.0)  # 摄像机看向 (0, 0, 2)
        self.gym.viewer_camera_look_at(self.viewer, self.env, cam_pos, cam_target)

        # 预处理仿真
        self.gym.prepare_sim(self.sim)

    def compute_observations(self):

        self.gym.refresh_dof_state_tensor(self.sim) #从仿真器获取数据

        env_ids = 1
        self.obs_buf[env_ids, 0] = self.dof_pos[env_ids, 0].squeeze() #小车位置
        self.obs_buf[env_ids, 1] = self.dof_vel[env_ids, 0].squeeze() #小车速度
        self.obs_buf[env_ids, 2] = self.dof_pos[env_ids, 1].squeeze() #倒立摆角度
        self.obs_buf[env_ids, 3] = self.dof_vel[env_ids, 1].squeeze() #倒立摆角速度
        self.obs_buf[env_ids, 4] = self.commands[env_ids].squeeze()   #新增观测

        pole_angle = self.obs_buf[:, 2]
        pole_vel   = self.obs_buf[:, 3]
        cart_vel   = self.obs_buf[:, 1]
        cart_pos   = self.obs_buf[:, 0]
        command    = self.obs_buf[:, 4]

        print(pole_angle)


    def step(self):
        """运行一步仿真，并更新 GUI"""
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        # 更新渲染（可视化）
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)
        self.gym.sync_frame_time(self.sim)

        self.compute_observations()


    def run_simulation(self, num_steps=1000,loop=1):
        """运行仿真循环"""
        if loop:
            while 1:
                if self.gym.query_viewer_has_closed(self.viewer):
                    break
                self.step()

        else:
            for _ in range(num_steps):
                if self.gym.query_viewer_has_closed(self.viewer):
                    break
                self.step()
        print("仿真结束")

    def close(self):
        """关闭仿真"""
        if self.viewer:
            self.gym.destroy_viewer(self.viewer)
        del self.gym
        print("仿真已关闭")


if __name__ == "__main__":
    # 运行仿真（带 GUI）
    env = CartPoleEnv(urdf_path=asset_root, urdf_file=asset_file)
    env.run_simulation(1000,1)
    env.close()
