# -*- coding: utf-8 -*-
import os, inspect
import numpy as np
import pybullet as p
import time
import pybullet_data
from liegroups.numpy import SO3
from pyb_manipulator.robots.jaco import Jaco
def create_environment():
    plane = [p.loadURDF("plane.urdf", 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000)]  # load plane
    objects = [p.loadURDF("table/table.urdf", 1.000000, -0.200000, 0.000000, 0.000000, 0.000000, 0.707107, 0.707107)]

def update_keys():
    keys = p.getKeyboardEvents()
    result = np.zeros(6)
    for k in keys:
        if k == ord('l'):
            result = np.array([0.1, 0, 0, 0, 0, 0])
        elif k == ord('j'):
            result = np.array([-0.1, 0, 0, 0, 0, 0])
        elif k == ord('i'):
            result = np.array([0, 0.1, 0, 0, 0, 0])
        elif k == ord('k'):
            result = np.array([0, -0.1, 0, 0, 0, 0])
        elif k == ord('o'):
            result = np.array([0, 0, 0.1, 0, 0, 0])
        elif k == ord('u'):
            result = np.array([0, 0, -0.1, 0, 0, 0])
        elif k == ord('e'):
            result = np.array([0, 0, 0, 0, 0, 2])
        elif k == ord('q'):
            result = np.array([0, 0, 0, 0, 0, -2])
    return result

cid = p.connect(p.GUI_SERVER)  # stuff above hangs much of the time for some reason
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))  # get current directory
p.resetSimulation()
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)
p.setGravity(0, 0, -10)
p.setRealTimeSimulation(1)

# Spawn the Jaco manipulator
create_environment()
jaco = Jaco(cid)

jaco.set_control_method('v') # we want velocity control
while (1):
    cmd = update_keys()
    pose = jaco.get_end_effector_pose()

    cmd[3:6] = SO3.from_quaternion(pose[3:7],'xyzw').dot(cmd[3:6]) # Transform from end effector frame to world
    jaco.set_end_effector_velocity(cmd, [0, 1, 2, 3, 4, 5])

    jaco.close_gripper()
    jaco.update()

p.disconnect()
