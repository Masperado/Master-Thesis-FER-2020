# -*- coding: utf-8 -*-
import os, inspect
import numpy as np
import pybullet as p
import time
import pybullet_data
from pyb_manipulator.utils import math_utils as mu
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
            result = np.array([0.05, 0, 0, 0, 0, 0])
        elif k == ord('j'):
            result = np.array([-0.05, 0, 0, 0, 0, 0])
        elif k == ord('i'):
            result = np.array([0, 0.05, 0, 0, 0, 0])
        elif k == ord('k'):
            result = np.array([0, -0.05, 0, 0, 0, 0])
        elif k == ord('o'):
            result = np.array([0, 0, 0.05, 0, 0, 0])
        elif k == ord('u'):
            result = np.array([0, 0, -0.05, 0, 0, 0])
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

# set up starting pose and goal
pose = jaco.get_end_effector_pose()
start_pos = pose[0:3]
start_rot = pose[3:7]
goal_pos = start_pos
goal_rot = start_rot

jaco.set_control_method('v') # we want velocity control
while (1):
    cmd = update_keys()
    pose = jaco.get_end_effector_pose()

    cur_pos = pose[0:3]
    cur_rot = pose[3:7]

    if(cmd.any()):
        cmd[3:6] = SO3.from_quaternion(cur_rot,'xyzw').dot(cmd[3:6])
        jaco.set_end_effector_velocity(cmd, [0, 1, 2, 3, 4, 5])
        goal_pos = pose[0:3]
        goal_rot = pose[3:7]
    else:
        w_t, w_r = mu.pose_error(cur_pos, cur_rot, goal_pos, goal_rot)
        jaco.set_end_effector_velocity(np.concatenate((w_t, w_r)), [0, 1, 2, 3, 4, 5])

    jaco.close_gripper()
    jaco.update()
    time.sleep(0.01)

p.disconnect()
