import mujoco
import mujoco.viewer

# model = mujoco.MjModel.from_xml_path("push.xml")
model = mujoco.MjModel.from_xml_path("panda_dual_push.xml")
data = mujoco.MjData(model)

mujoco.viewer.launch_passive(model, data)
input("Press Enter to close the viewer...")


#### mac崩溃
# import mujoco
# from mujoco import viewer

# model = mujoco.MjModel.from_xml_path("push.xml")
# data  = mujoco.MjData(model)

# with viewer.launch(model, data) as gui:
#     while gui.is_running():                      # GUI 内部在跑 mj_step
#         gui.sync()                               # 每帧把最新状态同步到窗口
