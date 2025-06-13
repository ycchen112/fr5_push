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

        self.object_name = "board" 
        self.slot_name = "board_site"
        self.notch_pos_offset = np.array([-0.2, 0, 0])

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
        mujoco.mj_resetData(self.model, self.data)
        if seed is not None:
            np.random.seed(seed)
        
        # 初始化板的位置
        # todo：初始化错
        object_pos = self.data.site(self.slot_name).xpos + np.random.uniform(-0.1, 0.1, 3)
        self.data.body(self.object_name).xpos = object_pos
        mujoco.mj_forward(self.model, self.data)

        return self.observation, {}
    
    def render(self, mode="human"):
        if mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            else:
                self.viewer.sync()

            object_pos = self.data.body(self.object_name).xpos
            notch_pos = object_pos + self.notch_pos_offset      # 缺口中心位置 
            slot_pos = self.data.site(self.slot_name).xpos
            dist_to_slot = np.linalg.norm(object_pos - slot_pos)
            print(f"Board pos: {object_pos}, Notch pos: {notch_pos}, \
                  Slot pos: {slot_pos}, Distance: {dist_to_slot:.3f}")
        return 
    
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def _get_obs(self):
        # 获取机械臂关节位置速度
        joint_pos = self.data.qpos[:6]
        joint_vel = self.data.qvel[:6]

        object_pos = self.data.body(self.object_name).xpos      # 板的位置
        object_quat = self.data.body(self.object_name).xquat    # 板的姿态
        slot_pos = self.data.site(self.slot_name).xpos          # 目标位置

        return np.concatenate([joint_pos, joint_vel, object_pos, object_quat, slot_pos])

    def _comput_reward(self):
        # todo：换计算方法
        object_pos = self.data.body(self.object_name).xpos
        slot_pos = self.datqa.body(self.slot_name).xpos
        dist_to_slot = np.linalg.norm(object_pos - slot_pos)
        reward = -dist_to_slot
        if dist_to_slot < 0.01:
            reward += 100.0
        action_penalty = -0.01 * np.sum(np.square(self.data.ctrl))

        return reward + action_penalty
    
    def _is_success(self):
        # todo: 验证能不能作为成果条件
        object_pos = self.data.body(self.object_name).xpos
        slot_pos = self.data.site(self.slot_name).xpos

        return np.linalg.norm(object_pos - slot_pos) < 0.01



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