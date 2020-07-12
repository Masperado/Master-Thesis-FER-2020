# -*- coding: utf-8 -*-
import os, inspect
import numpy as np
import pybullet as p
import time
import pybullet_data
from liegroups.numpy import SO3
from pyb_manipulator.robots.ur10 import Ur10

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
            result = np.array([0, 0, 0, 2, 0, 0])
        elif k == ord('q'):
            result = np.array([0, 0, 0, -2, 0, 0])
    return result

cid = p.connect(p.GUI)  # stuff above hangs much of the time for some reason
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))  # get current directory
p.resetSimulation()
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)
p.setGravity(0, 0, -10) # this is tricky, if you don't have a regulator set up the arm will drop (except in pos control)
p.setRealTimeSimulation(1)

# Spawn the UR10 manipulator
create_environment()
ur10 = Ur10(cid)

ur10.set_control_method('v') # we want velocity control
while (1):
    cmd = update_keys()
    pose = ur10.get_end_effector_pose()
    print(ur10.get_end_effector_velocity())
    cmd[3:6] = SO3.from_quaternion(pose[3:7],'xyzw').dot(cmd[3:6])
    ur10.set_end_effector_velocity(cmd, [0, 1, 2, 3, 4, 5])

    #m.close_gripper()
    ur10.update()

p.disconnect()