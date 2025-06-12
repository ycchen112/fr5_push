import os
import mujoco
import mujoco.viewer
import gymnasium as gym
import numpy as np
from gymnasium import spaces

current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, "..", "panda_description", "push.xml")
model_path = os.path.abspath(model_path)

class fr5_env(gym.Env):
    def __init__(self):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # 动作空间 6关节
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        # 观测空间 6关节角度 + 6关节速度 + 3物体位置 + 4物体姿态 + 3卡空位置
        self.observation_space = spaces.Box(low=-np.inf, high=-np.inf, shape=(21,), dtype=np.float32)
        self.max_step = 100
        self.current_step = 0
        self.viewer = None

    def step(self, action):
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        self.current_step += 1
        done = self.current_step >= self.max_step
        reward = 0.1

        obs = self._get_obs()

        return obs, reward, done, {}

    def reset(self, *, seed = None, options = None):
        self.current_step = 0
        self.observation = self.data.qpos.flatten()
        return self.observation, {}
    
    def render(self, mode="human"):
        if mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            else:
                self.viewer.sync()
        return 
    
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def _get_obs(self):
        # 获取机械臂关节位置速度
        joint_pos = self.data.qpos[:6]
        joint_vel = self.data.qvel[:6]

    def _comput_reward():
        

        return reward



if __name__ == "__main__":
    import time

    env = fr5_env()
    obs, info = env.reset()
    print("Initial observation:", obs)

    for i in range(1000):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"Step {i+1}: reward={reward}, done={done}")
        env.render()
        time.sleep(0.1)
        if done:
            break

    env.close()