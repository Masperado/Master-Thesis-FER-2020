from pyb_manipulator.manipulator import Manipulator
import os, inspect
import time

UR10_EE_IND = 8

class Ur10(Manipulator):

    def __init__(self, cid):
        currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))  # get current directory

        # Spawn the UR10 manipulator
        self.base_pos = [1, 0, 0.9]
        self.base_rot = [0, 0, 0, 1]  # p.getQuaternionFromEuler([0, 0, 0])

        super(Ur10, self).__init__(currentdir + "/../models/urdf/ur10.urdf", cid, UR10_EE_IND, 'p', [], self.base_pos, self.base_rot)

        jnt_goal = [0, -3.14 / 2, 3.14 / 2, -3.14 / 2, -3.14 / 2, 0]  # set joint position goal
        self.set_joint_position_goal(jnt_goal)
        self.update()
        time.sleep(1)

    def get_end_effector_pose(self):
        return self.get_link_pose(UR10_EE_IND)

    def get_end_effector_velocity(self):
        return self.get_link_vel(UR10_EE_IND)

    def set_end_effector_velocity(self, cmd, task):
        self.set_frame_velocity_goal(UR10_EE_IND, cmd,task)