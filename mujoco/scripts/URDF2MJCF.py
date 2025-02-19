import mujoco
from mujoco import mjcf_converter

urdf_path = "cartpole.urdf"  # 替换为你的 URDF 文件路径
mujoco_model = mjcf_converter.from_urdf(urdf_path)

# 保存为 MJCF（MuJoCo XML）
with open("your_robot.xml", "w") as f:
    f.write(mujoco_model.to_xml_string())
