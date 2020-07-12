import pybullet as p
from pyb_manipulator.utils import math_utils as mu
from numpy.linalg import lstsq, pinv, inv
from liegroups import SO3
import numpy as np

TASK_DIM = 6
JOINT_NAMES = 1
JOINT_ACTIVE = 3
LINK_NAMES = 12
ZERO_DISP = [0, 0, 0]
POS = range(0,3)
ROT = range(3,6)
KI = 0.01

class Manipulator(object):
    """
    Provides a pybullet API wrapper for simpler interfacing and manipulator-specific functions.
    The update() function should be called in a loop in order to store joint states and update joint controls.
    """

    def __init__(self, urdf_path, cid, ee_link_index, control_method, gripper_indices=(), base_pos=(0,0,0),
                 base_rot=(0, 0, 0, 1)):
        # user selected parameters -- non-private can be modified on the fly
        self._arm = [p.loadURDF(urdf_path)]  # arm object
        self._cid = cid  # process id
        self._num_jnt = p.getNumJoints(self._arm[0])  # number of joints
        self._num_lnk = p.getNumJoints(self._arm[0])  # Equal to the number of joints I think
        self._jnt_infos = [p.getJointInfo(self._arm[0], i) for i in range(self._num_jnt)]  # list of joint info objects

        self._active_ind = [j for j, i in zip(range(len(self._jnt_infos)), self._jnt_infos) if
                            i[JOINT_ACTIVE] > -1]  # indices of active joints
        self._gripper_ind = gripper_indices  # gripper join indices
        self._arm_ind = [e for e in self._active_ind if e not in tuple(self._gripper_ind)]  # arm joint indices

        self._num_jnt_gripper = len(self._gripper_ind)  # number of gripper joints
        self._num_jnt_arm = len(self._active_ind) - self._num_jnt_gripper  # number of arm joints

        self._control_method = control_method  # joint control method
        self._ee_link_ind = ee_link_index  # index of end effector link

        # define containers for states, poses, jacobians
        self.lnk_state = [None] * self._num_lnk
        self.lnk_pose = [None] * self._num_lnk
        self.lnk_vel = [None] * self._num_lnk
        self.J = np.zeros([self._num_lnk, TASK_DIM, self._num_jnt_arm])
        self.H = np.zeros([self._num_lnk, TASK_DIM, self._num_jnt_arm, self._num_jnt_arm])

        # set starting base position and orientation
        p.resetBasePositionAndOrientation(self._arm[0], base_pos, base_rot)

        self.get_joint_states()
        self._reset_all_flags()  # reset all flags

        # error used in I PID component
        self._e = 0

    def _reset_all_flags(self):
        """
        Reset all flags to false
        """
        self.__have_state = [False] * self._num_lnk
        self.__have_pose = [False] * self._num_lnk
        self.__have_vel = [False] * self._num_lnk
        self.__have_J = [False] * self._num_lnk
        self.__have_H = [False] * self._num_lnk

    # GET - PRIVATE
    # --------------------------------------------------------------------------------------------------------------

    def get_link_names(self):
        """
        Returns a list of all link names
        """
        names = []
        for info in self._jnt_infos:
            names.append(info[LINK_NAMES])

        return names

    def get_joint_names(self):
        """
        Returns a list of all joint names
        """
        names = []
        for info in self._jnt_infos:
            names.append(info[JOINT_NAMES])

        return names

    def get_joint_states(self):
        """
        Get positions, velocities and torques of active joints (as opposed to passive, fixed joints)
        """
        jnt_states = p.getJointStates(self._arm[0], range(p.getNumJoints(self._arm[0])))
        jnt_states = [j for j, i in zip(jnt_states, self._jnt_infos) if i[3] > -1]  # get only active states
        self.jnt_pos = np.array([state[0] for state in jnt_states])
        self.jnt_vel = np.array([state[1] for state in jnt_states])
        self.jnt_torq = np.array([state[3] for state in jnt_states])

        return self.jnt_pos, self.jnt_vel, self.jnt_torq

    def get_link_state(self, link_index):
        """
        Returns information on the link URDF frame and centre of mass poses in the world frame
        """
        if not self.__have_state[link_index]:
            self.lnk_state[link_index] = p.getLinkState(self._arm[0],
                                                        linkIndex=link_index, computeLinkVelocity=1)
            self.__have_state[link_index] = True

        return self.lnk_state[link_index]

    def get_link_pose(self, link_index):
        """
        Get a links pose in the world frame as a 7 dimensional vector containing the
        position (x,y,z) and quaternion (x,y,z,w)
        """
        if not self.__have_pose[link_index]:
            lnk_state = self.get_link_state(link_index)
            lnk_frame_pos = np.asarray(lnk_state[4])
            lnk_frame_rot = np.asarray(lnk_state[5])
            self.lnk_pose[link_index] = np.concatenate(
                (lnk_frame_pos, lnk_frame_rot))  # transform from x,y,z,w to w,x,y,z
            self.__have_pose[link_index] = True

        return self.lnk_pose[link_index]

    def get_link_vel(self, link_index, ref_frame_index = None):
        """
        Get a link's velocity in the given reference frame as a 6 dimensional vector containing
        translational and rotational velocity.
        :param link_index:
        :return:
        """
        if not self.__have_vel[link_index]:
            lnk_state = self.get_link_state(link_index)
            lnk_frame_lin_vel = np.asarray(lnk_state[6])
            lnk_frame_rot_vel = np.asarray(lnk_state[7])
            if ref_frame_index is not None:
                cur_rot = self.get_link_pose(ref_frame_index)[3:]
                lnk_frame_lin_vel = lnk_frame_lin_vel.dot(SO3.from_quaternion(cur_rot, 'xyzw').as_matrix())
                lnk_frame_rot_vel = lnk_frame_rot_vel.dot(SO3.from_quaternion(cur_rot, 'xyzw').as_matrix())

            self.lnk_vel[link_index] = np.concatenate((lnk_frame_lin_vel, lnk_frame_rot_vel))
            self.__have_vel[link_index] = True

        return self.lnk_vel[link_index]


    def get_link_acc(self):
        """
        Get a link's acceleration in the given reference frame as a 6 dimensional vector containing
        translational and rotational acceleration.
        :param link_index:
        :return:
        """
        raise NotImplementedError("Getting accelerations not yet implemented.")

    def _get_link_jacobian(self, link_index):
        """
        Get the Jacobian of a link frame in the form 6xN [J_trans; J_rot]
        """
        if not self.__have_J[link_index]:
            j_t, j_r = p.calculateJacobian(self._arm[0], link_index, ZERO_DISP, list(self.jnt_pos),
                                           [0] * len(self.jnt_pos), [0] * len(self.jnt_pos))
            j = np.concatenate((j_t, j_r), axis=0)
            self.J[link_index, :, :] = j[:, :self._num_jnt_arm]  # we don't need columns associated with the gripper
            self.__have_J[link_index] = True

        return self.J[link_index, :, :]

    def _get_link_hessian(self, link_index):
        """
        Compute the Jacobian derivative w.r.t joint angles
        Ref: Arjang Hourtash, 2005.
        """
        if not self.__have_H[link_index]:
            j = self._get_link_jacobian(link_index)

            for k in range(1, self._num_jnt_arm):
                j_k = j[:, k]
                for l in range(1, self._num_jnt_arm):
                    j_l = j[:, l]

                    h = (np.cross(j_k[ROT], j_l[POS]), np.cross(j_k[ROT], j_l[ROT]))
                    self.H[link_index, :, l, k] = np.concatenate(h, axis=0).T

            self.__have_H[link_index] = True

        return self.H[link_index, :, :, :]

    # JOINT CONTROL - PRIVATE
    # --------------------------------------------------------------------------------------------------------------

    def _hard_set_joint_positions(self, cmd):
        """
        Set joint positions without simulating actual control loops
        """
        k = 0
        cmd_ind = [j for j, i in zip(range(p.getNumJoints(self._arm[0])), self._jnt_infos) if i[3] > -1]
        for j in cmd_ind:
            p.resetJointState(self._arm[0], j, cmd[k])
            k = k + 1

    def _joint_position_control(self, cmd):
        """
        Position control of joints.
        """
        p.setJointMotorControlArray(self._arm[0], jointIndices=self._active_ind,
                                    controlMode=p.POSITION_CONTROL, targetPositions=cmd)

    def _joint_velocity_control(self, cmd):
        """
        Velocity control of joints. Uses PI regulator to remove steady state error.
        """
        self._e = self._e + (cmd-self.jnt_vel) # integrate error
        p.setJointMotorControlArray(self._arm[0], jointIndices=self._active_ind,
                                    controlMode=p.VELOCITY_CONTROL, targetVelocities= cmd + KI*self._e)

    # OTHER
    # ----------------------------------------------------------------------------------------------------------------

    def check_contact(self, objects=()):
        """
        Checks for contacts between the manipulator and given list of links indices.
        """
        if not objects:
            objects = range(self._num_jnt)

        for i in objects:
            cont = p.getContactPoints(self._arm[0], -1, i)
            if cont:
                return True

        return False

    # Manipulability stuff
    # ----------------------------------------------------------------------------------------------------------------
    def get_manipulability_ellipsoid(self, method, task):
        """
        Get manipulability ellipsoid of the end effector
        """
        j = self._get_link_jacobian(self._ee_link_ind)

        if method == 'force':
            return j[task, :].dot(j[task, :].T)
        elif method == 'velocity':
            return inv(j[task, :].dot(j[task, :].T))

    def get_manipulability_ellipsoid_jacobian(self, type, task):
        h = self._get_link_hessian(self._ee_link_ind)
        j = self._get_link_jacobian(self._ee_link_ind)

        jm = mu.mode_dot(h[task, :, :], j[task, :], 1) + mu.mode_dot(np.transpose(h[task, :, :], (1, 0, 2)), j[task, :],
                                                                     0)
        return jm

    # SET GOALS
    # ----------------------------------------------------------------------------------------------------------------

    def set_control_method(self, m):
        """
        Sets the control method variable
        """
        self._control_method = m

    def set_joint_position_goal(self, cmd):
        """
        Set goal joint position
        """
        self.pos_cmd = cmd

    def set_joint_velocity_goal(self, cmd):
        """
        Set goal joint velocity
        """
        self.vel_cmd = cmd

    def set_frame_pose_goal(self, index, t_pos, t_rot):
        ''' set a pose goal for an arbitrary frame'''
        result = p.calculateInverseKinematics(self._arm[0], index, targetPosition=t_pos.tolist(),
                                              targetOrientation=t_rot.tolist(), maxNumIterations=200,
                                              residualThreshold=0.002)

        help = np.array(result)
        self.set_joint_position_goal(np.concatenate((help[:6], np.zeros(self._num_jnt_gripper))))

    def set_frame_velocity_goal(self, index, t_vel, task):
        """
        Set Cartesian velocity goal for arbitrary frame
        """
        j = self._get_link_jacobian(index)

        dq, res, rank, a = lstsq(j[task, :],t_vel[task],rcond = None) # LS solver

        self.set_joint_velocity_goal(np.concatenate((dq, np.zeros(self._num_jnt_gripper))))  # Add zeros for gripper

    def set_vel_and_mnp_goal(self, index, t_vel, t_M, task1, task2):
        """
        Set Cartesian velocity and manipulability ellipsoid goals
        """
        j = self._get_link_jacobian(index)

        m = self.get_manipulability_ellipsoid('force', task2)  # current ME
        em = np.squeeze(np.array(mu.SPD_error(m, t_M).reshape(-1)))  # Look into Mandel notation

        jm = mu.unfold(self.get_manipulability_ellipsoid_jacobian('force', task2), 2) # ME Jacobian

        j_null = np.identity(self._num_jnt_arm) - pinv(j)[:,task1].dot(j[task1, :]) # nullspace

        #dq1, res1, rank1, a1 = lstsq(j[task1, :], t_vel[task1], rcond = None)  # LS solver
        #dq2, res2, rank2, a2 = lstsq((jm.T).dot(np.linalg.inv(j_null)), em, rcond = None )  # LS solver

        A = np.concatenate((pinv(j[task1,:]), j_null.dot(pinv(jm).T) ), axis = 1)
        #A = np.concatenate((j[task1,:], jm.T.dot( pinv(j_null).dot(pinv(j_null)) )))
        b = np.concatenate((t_vel[task1], em))
        dq, res, rank, a = lstsq(pinv(A), b, rcond = 0.001)

        #self.vel_cmd = np.concatenate((dq1 + dq2, np.zeros(self._num_jnt_gripper)))
        self.vel_cmd = np.concatenate((dq, np.zeros(self._num_jnt_gripper)))

    def close_gripper(self):
        """
        Close the robot gripper (modifies the current joint position command)
        """
        raise NotImplementedError("Implement a specific gripper close method.")


    def open_gripper(self):
        """
        Open the robot gripper (modifies the current joint position command)
        """
        raise NotImplementedError("Implement a specific gripper open method.")



    # UPDATE INTERNALLY
    # ----------------------------------------------------------------------------------------------------------------
    def update(self):
        """
        This function should be configurable
        """

        # get joint positions, velocities, torques
        self.get_joint_states()

        # run iteration of control loop
        if self._control_method == 'p':
            self._joint_position_control(self.pos_cmd)
        elif self._control_method == 'v':
            self._joint_velocity_control(self.vel_cmd)


        self._reset_all_flags()
