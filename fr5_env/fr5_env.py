import mujoco
import mujoco.viewer
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3 import spaces
from stable_baselines3.common.env_checker import check_env


class fr5_env(gym.Env):
    def __init__(self):
        super().__init__()

        self.model = mujoco.MjModel.from_xml_path("fr5_description/fr5.xml")
        self.data = mujoco.MjData(self.model)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=-np.inf, shape=(59,), dtype=np.float32)

        self.max_step = 100
        self.current_step = 0

    def step(self, action):


        return 

    def reset(self, *, seed = None, options = None):
        info = {}

        return self.observation, info