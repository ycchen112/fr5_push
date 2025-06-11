import os
import mujoco
import mujoco.viewer
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from gymnasium import spaces

current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, "..", "fr5_description", "urdf", "test.xml")
model_path = os.path.abspath(model_path)

class fr5_env(gym.Env):
    def __init__(self):
        super().__init__()

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=-np.inf, shape=(59,), dtype=np.float32)
        self.max_step = 100
        self.current_step = 0
        self.observation = np.zeros(59, dtype=np.float32)

    def step(self, action):
        mujoco.mj_step(self.model, self.data)


        done = self.current_step >= self.max_step
        reward = 0.0

        return self.observation, reward, done, {}

    def reset(self, *, seed = None, options = None):
        self.current_step = 0
        self.observation = self.data.qpos.flatten()  # Example: replace with actual observation extraction
        return self.observation, {}
    
    def render(self, mode="human"):
        if mode == "human":
            mujoco.viewer.launch_passive(self.model, self.data)
        return 
    
    def close(self):
        mujoco.mj_deleteModel(self.model)
        mujoco.mj_deleteData(self.data)

if __name__ == "__main__":
    import time

    env = fr5_env()
    obs, info = env.reset()
    print("Initial observation:", obs)

    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"Step {i+1}: reward={reward}, done={done}")
        env.render()
        time.sleep(0.1)
        if done:
            break

    env.close()