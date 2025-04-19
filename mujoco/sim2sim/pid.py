import numpy as np

class PIDController:
    def __init__(self, kp, ki, kd, derivative_filter_alpha=0,output_limit=None, dt=0.001):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.output_limit = output_limit  # 限幅，例如 10 表示 ±10

        self.integral = 0
        self.last_error = 0
        self.last_derivative = 0

        # 微分低通滤波器参数
        self.derivative_filter_alpha = derivative_filter_alpha

    def reset(self):
        self.integral = 0
        self.last_error = 0
        self.last_derivative = 0

    def compute(self, setpoint, feedback):
        error = setpoint - feedback

        self.integral += error * self.dt
        raw_derivative = (error - self.last_error) / self.dt

        # 应用一阶低通滤波器到微分项
        derivative = (
            self.derivative_filter_alpha * self.last_derivative +
            (1 - self.derivative_filter_alpha) * raw_derivative
        )
        self.last_derivative = derivative
        self.last_error = error

        output = self.kp * error + self.ki * self.integral + self.kd * derivative

        # 限幅
        if self.output_limit is not None:
            output = np.clip(output, -self.output_limit, self.output_limit)

        return output
