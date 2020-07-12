import os, inspect
import numpy as np
import pybullet as p
import time
import pybullet_data
from manipulator import Manipulator


def create_manipulator(urdf_path, ee_ind, base_pos, base_rot):
    jaco = [p.loadURDF(urdf_path)]  # load arm
    p.resetBasePositionAndOrientation(jaco[0], base_pos, base_rot)
    manipulator = Manipulator(jaco, cid, ee_ind, 'p')
    return manipulator


def create_plane(range):
    # set plane limits
    X_MAX = 0.2
    X_MIN = -0.2
    Y_MAX = 0.2
    Y_MIN = -0.2
    DENSITY = 0.05

    # choose normal
    v = np.random.normal(size=3)
    v = v / np.sqrt(np.sum(v ** 2))

    # choose points on 2d grid xy
    xx, yy = np.meshgrid(np.arange(range[0] + pose[0], range[1] + pose[0], DENSITY),
                         np.arange(range[2] + pose[1], range[3] + pose[1], DENSITY))
    xx = xx.flatten()
    yy = yy.flatten()

    # make that a plane
    d = -(np.dot(v, pose[0:3]))
    z = (-v[0] * xx - v[1] * yy - d) / v[2]

    # filter out anything outside of a given radius
    filter = (xx - pose[0]) ** 2 + (yy - pose[1]) ** 2 + (z - pose[2]) ** 2 < 0.3 ** 2
    xx = xx[filter]
    yy = yy[filter]
    z = z[filter]
    return xx, yy, z


cid = p.connect(p.GUI_SERVER)  # stuff above hangs much of the time for some reason
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))  # get current directory

# start simulation
p.resetSimulation()
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
p.setGravity(0, 0, -10)
p.setRealTimeSimulation(1)

m_base_pos = [1, 0, 0.7]
m_base_rot = p.getQuaternionFromEuler([0, 0, 0])
m_ee_ind = 7
m = create_manipulator(currentdir + "/models/urdf/jaco.urdf", m_ee_ind, m_base_pos, m_base_rot)

m_jnt_pos_start = [0, 3.14, 3.14 / 2, -3.14 / 2, -3.14 / 2, 0, 0, 0, 0]  # set joint position start
m.set_joint_position_goal(m_jnt_pos_start)

m.update()  # update joint position
time.sleep(1)  # give it some time to set

pose = m.get_end_effector_pose()

x, y, z = create_plane([-0.2, 0.2, -0.2, 0.2])

for xi, yi, zi in zip(x, y, z):
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.01, visualFramePosition=[0, 0, 0], rgbaColor=[1, 0, 0, 1])
    test_body = p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, basePosition=[xi, yi, zi])
    print(m.set_frame_pose_goal(m_ee_ind, np.array([xi, yi, zi]), pose[3:7])) # set goal with fixed orientation and changing positon
    m.update()
    time.sleep(1)

while (1):
    time.sleep(0.01)
