import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("push.xml")
data = mujoco.MjData(model)

mujoco.viewer.launch_passive(model, data)
input("Press Enter to close the viewer...")
