import mujoco
import mujoco.viewer

model_path = "scene.xml"

model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

print("Bodies:", model.nbody)
print("Joints:", model.njnt)
print("Actuators:", model.nu)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()