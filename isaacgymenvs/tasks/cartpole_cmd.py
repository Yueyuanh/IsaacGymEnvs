# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import os
import torch
from isaacgymenvs.utils.torch_jit_utils import to_torch, get_axis_params, torch_rand_float, quat_rotate, quat_rotate_inverse
from isaacgym import gymutil, gymtorch, gymapi
from .base.vec_task import VecTask
from omegaconf import OmegaConf

class CartpoleCmd(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.reset_dist = self.cfg["env"]["resetDist"]

        self.max_push_effort = self.cfg["env"]["maxEffort"]
        self.max_episode_length = 500

        self.cfg["env"]["numObservations"] = 5
        self.cfg["env"]["numActions"] = 1


        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

        # 新增随机位置控制
        self.command_pos_range = self.cfg["env"]["randomCommandPosRanges"]#控制指令
        self.commands = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)#增加控制指令

        # 读取环境变量play
        current_dir = os.path.dirname(os.path.realpath(__file__))
        config_path = os.path.join(current_dir, '..', 'cfg', 'play_config.yaml')
        self.config=OmegaConf.load(config_path)
        print(self.config.test)


    def create_sim(self):
        # set the up axis to be z-up given that assets are y-up by default
        self.up_axis = self.cfg["sim"]["up_axis"]

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))# 每行要放置的数量

    # 创建第平面
    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        # set the normal force to be z dimension
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0) if self.up_axis == 'z' else gymapi.Vec3(0.0, 1.0, 0.0)
        self.gym.add_ground(self.sim, plane_params)

    # 创建环境
    def _create_envs(self, num_envs, spacing, num_per_row): # 每行要放置的数量
        # define plane on which environments are initialized
        # 单个环境 spacing:每个环境的边长
        lower = gymapi.Vec3(0.5 * -spacing, -spacing, 0.0) if self.up_axis == 'z' else gymapi.Vec3(0.5 * -spacing, 0.0, -spacing)
        upper = gymapi.Vec3(0.5 * spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        asset_file = "urdf/cartpole_cmd.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        # 导入资产额外信息
        asset_options = gymapi.AssetOptions() # 额外选项
        asset_options.fix_base_link = True # 固定base_link
        cartpole_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(cartpole_asset)
 
        # 加载资产并旋转
        pose = gymapi.Transform()
        if self.up_axis == 'z':
            pose.p.z = 2.0
            # asset is rotated z-up by default, no additional rotations needed
            pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        else:
            pose.p.y = 2.0
            pose.r = gymapi.Quat(-np.sqrt(2)/2, 0.0, 0.0, np.sqrt(2)/2)

        self.cartpole_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            # 资产创建Actor
            cartpole_handle = self.gym.create_actor(env_ptr, cartpole_asset, pose, "cartpole", i, 1, 0)

            dof_props = self.gym.get_actor_dof_properties(env_ptr, cartpole_handle)
            dof_props['driveMode'][0] = gymapi.DOF_MODE_EFFORT #扭矩模式
            dof_props['driveMode'][1] = gymapi.DOF_MODE_NONE
            dof_props['stiffness'][:] = 0.0
            dof_props['damping'][:] = 0.0
            self.gym.set_actor_dof_properties(env_ptr, cartpole_handle, dof_props)

            self.envs.append(env_ptr)
            self.cartpole_handles.append(cartpole_handle)

    def compute_reward(self): #计算奖励
        # retrieve environment observations from buffer
        pole_angle = self.obs_buf[:, 2]
        pole_vel   = self.obs_buf[:, 3]
        cart_vel   = self.obs_buf[:, 1]
        cart_pos   = self.obs_buf[:, 0]
        command    = self.obs_buf[:, 4]

        self.rew_buf[:], self.reset_buf[:] = compute_cartpole_reward(
            command,
            pole_angle, pole_vel, cart_vel, cart_pos,
            self.reset_dist, self.reset_buf, self.progress_buf, self.max_episode_length
        )

    def compute_observations(self, env_ids=None): #观测器
        if env_ids is None:
            env_ids = np.arange(self.num_envs)

        self.gym.refresh_dof_state_tensor(self.sim) #从仿真器获取数据

        self.obs_buf[env_ids, 0] = self.dof_pos[env_ids, 0].squeeze() #小车位置
        self.obs_buf[env_ids, 1] = self.dof_vel[env_ids, 0].squeeze() #小车速度
        self.obs_buf[env_ids, 2] = self.dof_pos[env_ids, 1].squeeze() #倒立摆角度
        self.obs_buf[env_ids, 3] = self.dof_vel[env_ids, 1].squeeze() #倒立摆角速度
        self.obs_buf[env_ids, 4] = self.commands[env_ids].squeeze()   #新增观测

        return self.obs_buf

    def reset_idx(self, env_ids): #复位
        positions  = 0.2 * (torch.rand((len(env_ids), self.num_dof), device=self.device) - 0.5)
        velocities = 0.5 * (torch.rand((len(env_ids), self.num_dof), device=self.device) - 0.5)

        self.dof_pos[env_ids, :] = positions[:]
        self.dof_vel[env_ids, :] = velocities[:]

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        

        # todo: play remote control
        print("*****************playing******************")
        print(self.config.test)

        #随机期望位置
        self.commands[env_ids] = torch_rand_float(-self.command_pos_range , self.command_pos_range, (len(env_ids), 1), device=self.device)


        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0



    def pre_physics_step(self, actions): #预物理步骤
        actions_tensor = torch.zeros(self.num_envs * self.num_dof, device=self.device, dtype=torch.float)
        actions_tensor[::self.num_dof] = actions.to(self.device).squeeze() * self.max_push_effort
        forces = gymtorch.unwrap_tensor(actions_tensor)
        self.gym.set_dof_actuation_force_tensor(self.sim, forces)


    def post_physics_step(self): 
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward()
        
        # debug viz
        self.gym.clear_lines(self.viewer)
        if 1:#self.viewer and self.debug_viz:
            for i in range(self.num_envs):
                #i=23
                #self.cfg["env"]['envSpacing']
                origin = self.gym.get_env_origin(self.envs[i])#<<-----------------获取空间原点
                location=(origin.x, self.commands[i]+origin.y , 2.0)
                color=(1, 1, 0)
                self.draw_sphere(location,color)
                #gymutil.draw_lines(sphere_geom, self.gym, self.viewer, 0, sphere_pose)

    def draw_sphere(self, location, color):
        #self.gym.clear_lines(self.viewer)
        sphere_geom = gymutil.WireframeSphereGeometry(0.1, 3, 3, None, color)
        pose = gymapi.Transform(gymapi.Vec3(location[0], location[1], location[2]), r=None)
        gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[0], pose)

#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
# 计算奖励：（设定位置,摆角度，摆角速度，车速度，车位置
#           复位初始距离，复位标志，，训练回合）
def compute_cartpole_reward(command,pole_angle, pole_vel, cart_vel, cart_pos,
                            reset_dist, reset_buf, progress_buf, max_episode_length):
    # type: (Tensor,Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]

    # reward is combo of angle deviated from upright, velocity of cart, and velocity of pole moving
    # reward = 1.0 - pole_angle * pole_angle - 0.01 * torch.abs(cart_vel) - 0.005 * torch.abs(pole_vel)
    reward = 1.0 - pole_angle * pole_angle*0.5 - 0.01 * torch.abs(cart_vel) - 0.01 * torch.abs(pole_vel) - torch.abs(command-cart_pos)*0.8
    # adjust reward for reset agents
    reward = torch.where(torch.abs(cart_pos) > reset_dist, torch.ones_like(reward) * -2.0, reward)
    reward = torch.where(torch.abs(pole_angle) > np.pi / 2, torch.ones_like(reward) * -2.0, reward)

    reset = torch.where(torch.abs(cart_pos) > reset_dist, torch.ones_like(reset_buf), reset_buf)
    reset = torch.where(torch.abs(pole_angle) > np.pi / 2, torch.ones_like(reset_buf), reset)
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset)

    return reward, reset
