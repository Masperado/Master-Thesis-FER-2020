from pyb_manipulator.manipulator import Manipulator
import numpy as np
import os, inspect
import time

JACO_EE_IND = 8
GRIPPER_MAX = [0.3, 1.33]
GRIPPER_VEL = 4

class Jaco(Manipulator):

    def __init__(self, cid):
        currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))  # get current directory

        # Spawn the UR10 manipulator
        self.base_pos = [1, 0, 0.9]
        self.base_rot = [0, 0, 0, 1]  # p.getQuaternionFromEuler([0, 0, 0])

        super(Jaco, self).__init__(currentdir + "/../models/urdf/jaco.urdf", cid, JACO_EE_IND, 'p', [9,11,13], self.base_pos, self.base_rot)

        jnt_goal = [0, 3.14, 3.14 / 2, -3.14 / 2, -3.14 / 2, 0, 0, 0, 0]  # set joint position goal
        self.set_joint_position_goal(jnt_goal)
        self.update()
        time.sleep(1)

    def get_end_effector_pose(self):
        return self.get_link_pose(JACO_EE_IND)

    def set_end_effector_velocity(self, cmd, task):
        self.set_frame_velocity_goal(JACO_EE_IND, cmd,task)

    def close_gripper(self):
        """
        Close the robot gripper (modifies the current joint position command)
        """
        if self._control_method == 'p':
            # TODO this is probably no good, instead should allow setting position directly
            self.pos_cmd[-self._num_jnt_gripper:] = GRIPPER_MAX[1] * np.ones(self._num_jnt_gripper)
        elif self._control_method == 'v':
            self.vel_cmd[-self._num_jnt_gripper:] = GRIPPER_VEL * (self.jnt_pos[-self._num_jnt_gripper:]
                                                     <= GRIPPER_MAX[1]).astype(float)

    def open_gripper(self):
        """
        Open the robot gripper (modifies the current joint position command)
        """
        if self._control_method == 'p':
            self.pos_cmd[-self._num_jnt_gripper:] = GRIPPER_MAX[0] * np.ones(self._num_jnt_gripper)
        elif self._control_method == 'v':
            self.vel_cmd[-self._num_jnt_gripper:] = -GRIPPER_VEL * (self.jnt_pos[-self._num_jnt_gripper:]
                                                                         - GRIPPER_MAX[0])
