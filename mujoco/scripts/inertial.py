def compute_box_inertia_xml(size, mass, pos=(0, 0, 0.5)):
    """
    计算均匀长方体的惯性，并生成 MuJoCo XML 格式的 <inertial> 标签
    :param size: (a, b, c)，box 的半尺寸
    :param mass: 总质量
    :param pos: 惯性中心的位置，默认设为 box 几何中心
    :return: str，XML 格式字符串
    """
    a, b, c = size
    px, py, pz = pos
    I_x = (1/3) * mass * (b**2 + c**2)
    I_y = (1/3) * mass * (a**2 + c**2)
    I_z = (1/3) * mass * (a**2 + b**2)
    
    xml = f'<inertial pos="{px} {py} {pz}" mass="{mass}" diaginertia="{I_x:.6f} {I_y:.6f} {I_z:.6f}"/>'
    return xml


# 示例参数
size = (0.02, 0.03, 0.5)   # 半尺寸
mass = 1.0
pos = (0, 0, 0.5)          # 中心点位置

# 输出 XML 标签
xml_output = compute_box_inertia_xml(size, mass, pos)
print(xml_output)
