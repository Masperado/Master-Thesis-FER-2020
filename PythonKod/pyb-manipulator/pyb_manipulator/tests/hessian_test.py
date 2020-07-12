# -*- coding: utf-8 -*-
import os, inspect
import numpy as np
import pybullet as p
import time
import pybullet_data
from manipulator import Manipulator
from numpy import linalg as la


def hessian1(m):
    """
    Compute the Jacobian derivative w.r.t joint angles (hybrid Jacobian
    representation).
    Ref: Arjang Hourtash, 2005.
    """
    J = m.get_end_effector_jacobian()
    H= np.zeros((6, m._num_jnt_arm, m._num_jnt_arm))

    for i in range(1, m._num_jnt_arm):
        J_i = J[:, i]
        for j in range(1, m._num_jnt_arm):
            J_j = J[:, j]

            h = (np.cross(J_i[3:6], J_j[0:3], axis=0), np.cross(J_i[3:6], J_j[3:6], axis=0))
            H[:, i,j] = np.concatenate(h,axis=0).T  # my hessian calculation might not be correct

    return H

def hessian2(m):
    """
     Compute the Jacobian derivative w.r.t joint angles (hybrid Jacobian
     representation).
     Ref: H. Bruyninck and J. de Schutter, 1996
     """
    J = m.get_end_effector_jacobian()
    H = np.zeros((6, m._num_jnt_arm, m._num_jnt_arm))

    for i in range(1, m._num_jnt_arm):
        J_i  = J[:, i]
        for j in range(1, m._num_jnt_arm):
            J_j = J[:,j]

            if(j < i):
                h = (np.cross(J_j[3:6], J_i[0:3], axis=0), np.cross(J_j[3:6], J_i[3:6], axis=0))
                H[ :, i, j] = np.concatenate(h, axis=0).T
            elif j > i:
                h = -np.cross(J_j[0:3], J_i[3:6], axis=0)
                H[ 0:3, i, j] = np.concatenate(h, axis=0).T
            else:
                h = np.cross(J_i[3:6], J_i[0:3], axis=0)
                H[ 0:3, i, j] = np.concatenate(h, axis=0).T #for some reason he only calculates half the matrix

    return H

def create_environment():
    plane = [p.loadURDF("plane.urdf", 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000)]  # load plane
    objects = [p.loadURDF("table/table.urdf", 1.000000, -0.200000, 0.000000, 0.000000, 0.000000, 0.707107, 0.707107)]

def update_keys():
    keys = p.getKeyboardEvents()
    result = np.array([])
    for k in keys:
        if k == ord('l'):
            result = np.array([0.05, 0, 0])
        elif k == ord('j'):
            result = np.array([-0.05, 0, 0])
        elif k == ord('i'):
            result = np.array([0, 0.05, 0])
        elif k == ord('k'):
            result = np.array([0, -0.05, 0])
        elif k == ord('o'):
            result = np.array([0, 0, 0.05])
        elif k == ord('u'):
            result = np.array([0, 0, -0.05])
        elif k == ord('u'):
            result = np.array([0, 0, -0.05])

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

# Spawn the Jaco manipulator
m_base_pos = [1, 0, 0.9]
m_base_rot = p.getQuaternionFromEuler([0, 0, 0])
m_ee_ind = 7
m = Manipulator(currentdir + "/../models/urdf/jaco.urdf", cid, m_ee_ind, 'p', [9,11,13], m_base_pos, m_base_rot)
create_environment()

jointPositions = [0, 3.14, 3.14 / 2, -3.14 / 2, -3.14 / 2, 0, 0, 0, 0]  # set joint position goal
m.set_joint_position_goal(jointPositions)
m.update()  # update joint position
time.sleep(1)

# set up starting pose and goal
pose = m.get_end_effector_pose()
start_pos = pose[0:3]
start_rot = pose[3:7]
goal_pos = start_pos
goal_rot = start_rot

m.set_control_method('v') # we want velocity control
M = m.get_manipulability_ellipsoid('force',[0,1,2,3,4,5])
while (1):
    cmd = update_keys()
    pose = m.get_end_effector_pose()
    cur_pos = pose[0:3]
    cur_rot = pose[3:7]
    if(cmd.any()):
        goal_pos = cur_pos + cmd
        #goal_rot = start_rot +
    w_t, w_r = m._pose_error(cur_pos, cur_rot, goal_pos, goal_rot)
    m.set_frame_velocity_goal(m_ee_ind , np.concatenate((w_t, w_r)), [0, 1, 2, 3, 4, 5])
    m.close_gripper()
    m.update()

    H1 = hessian1(m)
    H2 = hessian2(m)
    print(H1[:,3,5] - H2[:,5,3]) # H1[:,j,i] = H2[:,i,j]
    print('****')
    print(H1[:,2,1])
    print('-----')


p.disconnect()
