- [ ] # Isaac Gym

## NVIDIA ISAAC

​	在NVIDIA官网的IsaacGym安装包是Python中`isaacgym`的安装包，同时应当在本地建立conda环境(bash:rl)，方便环境管理，若在远程部署则可以使用docker完成快速部署训练。

​	除此之外，NVIDIA推出了新的机器人强化学习仿真平台，IsaacSim & IsaacLab，相较于IsaacGym，IS&IL有类似Unity的可视化的操作界面，方便资产管理与开发，同时能够更好的与其硬件做适配，也有更加丰富的功能，同时对硬件的要求也更高(>RTX 3070)。

​	尽管，IsaacGym已经逐渐落伍，但有前辈借助其完成了很多出色的任务，并以此打开了笔者学习强化学习的大门，其还是有很大的学习必要的，作为初学和了解强化学习的基本流程是很合适的（限于3050的老破小）。

## IsaacGymEnvs



​	于此同时，NVIDIA也开源了其基于IsaacGym的开发框架即`IsaacGymEnvs`，其中包含一些利用IsaacGym实现机器人强化学习训练的丰富例程，都是采用PPO进行训练，以供学习者进行开发、实验。

​	为了方便管理学习记录和开发流程，在自己的Github下fork了官方的IsaacGymEnvs（Github：https://github.com/Yueyuanh/IsaacGymEnvs.git）。

# IsaacGymEnvs Note

## IsaacGymEnvs 代码框架

### 1.文件结构

.
├── `assets`  //资产文件:URDF、MJCF(mujoco) 
│   ├── asset_templates
│   ├── mjcf
│   │   └── open_ai_assets
│   │       ├── fetch
│   │       ├── hand
│   │       ├── stls
│   │       │   ├── fetch
│   │       │   └── hand
│   │       └── textures
│   ├── trifinger
│   └── `urdf`
│       ├── anymal_c
│       │   ├── meshes
│       │   └── urdf
│       ├── franka_description
│       ├── kuka_allegro_description
│       ├── sektion_cabinet_model
│       ├── tray
│       └── ycb
├── docs  //官方文档
└── `isaacgymenvs`      //训练主环境
    ├── `cfg`		//配置文件
    │   ├── pbt
    │   ├── runs          //训练网络
    │   ├── task	  //训练环境配置文件 eg.Cartpole.ymal
    │   └── train 	//强化学习参数配置文件eg.CartpolePPO.yaml
    ├── learning	//？？？
    ├── pbt			//？？？
    ├── `tasks`		//训练任务代码：加载环境、配置奖励、训练过程
    │   ├── `__init__.py`	//训练任务注册
    │   ├── amp
    │   ├── base
    │   └── utils
    ├── utils    //其他模块
    ├── __init__.py    //命令arg初始化
    └── `train.py`       //训练代码

### 2.args 训练参数

Key arguments to the `train.py` script are:

* `task=TASK` - selects which task to use. Any of `AllegroHand`, `AllegroHandDextremeADR`, `AllegroHandDextremeManualDR`, `AllegroKukaLSTM`, `AllegroKukaTwoArmsLSTM`, `Ant`, `Anymal`, `AnymalTerrain`, `BallBalance`, `Cartpole`, `FrankaCabinet`, `Humanoid`, `Ingenuity` `Quadcopter`, `ShadowHand`, `ShadowHandOpenAI_FF`, `ShadowHandOpenAI_LSTM`, and `Trifinger` (these correspond to the config for each environment in the folder `isaacgymenvs/config/task`)
* `train=TRAIN` - selects which training config to use. Will automatically default to the correct config for the environment (ie. `<TASK>PPO`).
* `num_envs=NUM_ENVS` - selects the number of environments to use (overriding the default number of environments set in the task config).
* `seed=SEED` - sets a seed value for randomizations, and overrides the default seed set up in the task config
* `sim_device=SIM_DEVICE_TYPE` - Device used for physics simulation. Set to `cuda:0` (default) to use GPU and to `cpu` for CPU. Follows PyTorch-like device syntax.
* `rl_device=RL_DEVICE` - Which device / ID to use for the RL algorithm. Defaults to `cuda:0`, and also follows PyTorch-like device syntax.
* `graphics_device_id=GRAPHICS_DEVICE_ID` - Which Vulkan graphics device ID to use for rendering. Defaults to 0. **Note** - this may be different from CUDA device ID, and does **not** follow PyTorch-like device syntax.
* `pipeline=PIPELINE` - Which API pipeline to use. Defaults to `gpu`, can also set to `cpu`. When using the `gpu` pipeline, all data stays on the GPU and everything runs as fast as possible. When using the `cpu` pipeline, simulation can run on either CPU or GPU, depending on the `sim_device` setting, but a copy of the data is always made on the CPU at every step.
* `test=TEST`- If set to `True`, only runs inference on the policy and does not do any training.
* `checkpoint=CHECKPOINT_PATH` - Set to path to the checkpoint to load for training or testing.
* `headless=HEADLESS` - Whether to run in headless mode.
* `experiment=EXPERIMENT` - Sets the name of the experiment.
* `max_iterations=MAX_ITERATIONS` - Sets how many iterations to run for. Reasonable defaults are provided for the provided environments.

Hydra also allows setting variables inside config files directly as command line arguments. As an example, to set the discount rate for a rl_games training run, you can use `train.params.config.gamma=0.999`. Similarly, variables in task configs can also be set. For example, `task.env.enableDebugVis=True`.

