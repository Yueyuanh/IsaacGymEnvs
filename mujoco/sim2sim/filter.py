class LowPassFilter:
    def __init__(self, alpha, initial_value=0.0):
        """
        初始化低通滤波器
        :param alpha: 滤波系数，越接近 1 越平滑
        :param initial_value: 初始输出值
        """
        self.alpha = alpha
        self.prev_value = initial_value

    def reset(self, value=0.0):
        """重置滤波器状态"""
        self.prev_value = value

    def filter(self, new_value):
        """应用低通滤波器，并自动更新内部状态"""
        self.prev_value = self.alpha * self.prev_value + (1 - self.alpha) * new_value
        return self.prev_value
