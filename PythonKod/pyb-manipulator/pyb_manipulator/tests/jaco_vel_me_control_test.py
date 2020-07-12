# -*- coding: utf-8 -*-
import os, inspect
import numpy as np
import pybullet as p
import time
import matplotlib.pyplot as plt
import pybullet_data
from utils import math_utils as mu
from robots.jaco import Jaco

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


def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:" + fmt + "}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:" + str(col_maxes[i]) + fmt + "}").format(y), end="  ")
        print("")


cid = p.connect(p.GUI_SERVER)  # stuff above hangs much of the time for some reason
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))  # get current directory

p.resetSimulation()
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
p.setGravity(0, 0, -10)
p.setRealTimeSimulation(1)

create_environment()
jaco = Jaco(cid)

# set up starting pose and goal
pose = jaco.get_end_effector_pose()

# set up starting pose and goal
start_pos = pose[0:3]
start_rot = pose[3:7]
goal_pos = start_pos
goal_rot = start_rot
cur_pos = start_pos
cur_rot = start_rot
time.sleep(1)
time.gmtime()
jaco.set_control_method('v') # we want velocity control
M = jaco.get_manipulability_ellipsoid('force',[0,1,2])
#vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.01, visualFramePosition=[0, 0, 0], rgbaColor=[1, 0, 0, 1])
plt.ion() ## Note this correction
fig=plt.figure()
plt.interactive(False)
start_time = time.time()
while (1):
    cmd = update_keys()
    pose = jaco.get_end_effector_pose()
    cur_pos = pose[0:3]
    cur_rot = pose[3:7]
    if(cmd.any()):
        jaco.set_end_effector_velocity(cmd, [0, 1, 2])
        goal_pos = pose[0:3]
        goal_rot = pose[3:7]
    else:
        w_t, w_r = mu.pose_error(cur_pos, cur_rot, goal_pos, goal_rot)
        jaco.set_vel_and_mnp_goal(8, np.concatenate((w_t, w_r)), M, [0, 1, 2], [0, 1, 2])

    jaco.close_gripper()
    jaco.update()

    err = mu.SPD_error(jaco.get_manipulability_ellipsoid('force', [0, 1, 2]), M).reshape(-1)

    delta = time.time() - start_time
    plt.plot([delta, delta], [err[0,1], err[0,2]],'r.')
    plt.axis([0, time.time() - start_time +1, 0, 2])
    plt.show(False)
    plt.draw()
    plt.pause(0.0001)

p.disconnect()
