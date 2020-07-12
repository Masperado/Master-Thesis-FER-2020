import gym
import os, inspect
import numpy as np
import pybullet as p
import math
import time
import pybullet_data
from random import random, randint, choice
from liegroups.numpy import SO3
from pyb_manipulator.robots.jaco import Jaco


class MetakEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    target = np.array([0, 0, 0])
    obsTar = []
    obs_body = []
    distance_reward_old = 0
    # obs = False
    obs = True
    NUM_OBS = 3
    NUM_LINKS = 0
    # NUM_LINKS = 4

    def __init__(self, episodeLength=5000):
        # use the GUI_SERVER line if you want visual feedback
        # beware, the GUI_SERVER doesn't support multithreading
        # use the DIRECT line if you want to train on the environment

        self.cid = p.connect(p.GUI_SERVER)
        # self.cid = p.connect(p.DIRECT)

        # various pybullet configuration commands
        currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)
        p.setGravity(0, 0, -10)
        p.setRealTimeSimulation(0)

        # creating the environment and spawning Jaco
        self.plane = [p.loadURDF("plane.urdf", 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000)]
        self.targetSphere = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, visualFramePosition=[0, 0, 0],
                                                rgbaColor=[1, 0, 0, 1])
        self.target_body = p.createMultiBody(baseMass=0, baseVisualShapeIndex=self.targetSphere,
                                             basePosition=self.target)
        self.jaco = Jaco(self.cid)

        if (self.obs):
            self.obsSphereCol = p.createCollisionShape(p.GEOM_SPHERE, radius=0.1)
            self.obsSphereVis = p.createVisualShape(p.GEOM_SPHERE, radius=0.1,
                                                    visualFramePosition=[0, 0, 0], rgbaColor=[1, 1, 0, 1])
            for i in range(0, self.NUM_OBS):
                self.obsTar.append(np.array([0, 0, 0]))
                self.obs_body.append(0)
                self.obs_body[i] = p.createMultiBody(baseMass=0, baseVisualShapeIndex=self.obsSphereVis,
                                                     baseCollisionShapeIndex=self.obsSphereCol,
                                                     basePosition=self.obsTar[i])

        # setting the control method to velocity
        self.jaco.set_control_method('v')

        # the seed command isn't actually a seeder at this point
        # self.seed() generates a target inside the environment for the manipulator to reach
        self.seed()

        # set internal step counter
        self.numberOfSteps = 0

        # how many steps until the environment is reset 
        self.episodeLength = episodeLength

        # defining the action and observation spaces
        high = np.inf * np.ones([6])
        if (self.obs):
            high = np.inf * np.ones([6 + 3 * self.NUM_OBS + 3 * self.NUM_LINKS])

        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=np.array([-0.2, -0.2, -0.2, -0.2, -0.2, -0.2]),
                                           high=np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2]), dtype=np.float32)

    def step(self, action):
        self.numberOfSteps += 1

        # this should prevent a nonexistant action from breaking the code
        if 'action' not in locals():
            action = np.array([0, 0, 0, 0, 0, 0])

        # every recieved action gets clipped to the size of the action space
        # this is done to ensure compatibility with inf. action space algorithms (ex. PPO2)
        if not self.action_space.contains(action):
            action = np.clip(action, self.action_space.low, self.action_space.high)

        # conversion of the action array into an appropriate shape
        # and simulation stepping
        pose = self.jaco.get_end_effector_pose()
        action[3:6] = SO3.from_quaternion(pose[3:7], 'xyzw').dot(action[3:6])
        self.jaco.set_end_effector_velocity(action, [0, 1, 2, 3, 4, 5])
        self.jaco.close_gripper()
        self.jaco.update()
        p.stepSimulation()

        # reward calculation
        observation = self.get_observation()
        distance_vector = observation[:3] - self.target
        distance_reward = - np.linalg.norm(distance_vector)
        progress_reward = - 2 * 300 * (-distance_reward + self.distance_reward_old)
        self.distance_reward_old = distance_reward
        speed_reward = - np.linalg.norm(action)
        if (self.obs):
            min_obstacle_distances = self.min_obstacle_distances()
            obstacle_reward = -0.02 / (np.min(min_obstacle_distances) ** 3)
        reward = distance_reward + progress_reward + speed_reward
        if (self.obs):
            reward += obstacle_reward

        # check if the goal has been reached, otherwise check if the episode is over
        if np.linalg.norm(distance_vector) < 0.05:
            print('#### Goal', self.target, 'reached, with norm', np.linalg.norm(distance_vector), '. ####')
            done = True
        elif self.obs and np.any(min_obstacle_distances < 0.15):
            print('#### Hit obstacle. Reseting. ####')
            done = True
            # done = False
            time.sleep(2)
        elif self.numberOfSteps > self.episodeLength:
            print('#### Step limit reached, reseting... ####')
            done = True
        else:
            done = False

        # debug info
        info = dict(action=action, reward=reward)

        if (self.obs):
            return np.concatenate((self.target, observation[:3], *self.obsTar, *self.get_links()),
                                  0), reward, done, info
        else:
            return np.concatenate((self.target, observation[:3]), 0), reward, done, info

    def min_obstacle_distances(self):
        min_distances = np.inf * np.ones([1 + self.NUM_LINKS])
        for i in range(8 - self.NUM_LINKS, 9):
            observation = self.jaco.get_link_pose(i)[:3]
            for j in range(0, self.NUM_OBS):
                distance = np.linalg.norm(observation - self.obsTar[j])
                if (distance < min_distances[i - (8 - self.NUM_LINKS)]):
                    min_distances[i - (8 - self.NUM_LINKS)] = distance
        return min_distances

    def get_observation(self):
        return self.jaco.get_end_effector_pose()

    def get_links(self):
        poses = []
        for i in range(8 - self.NUM_LINKS, 8):
            poses.append(self.jaco.get_link_pose(i)[:3])
        return poses

    def seed(self, seed=None):
        # randomly select a point in the spere around Jaco
        # set it as the goal
        phi = random() * 2 * math.pi
        cosTheta = random() * 2 - 1
        u = random()

        theta = math.acos(cosTheta)
        r = 0.3 * (u ** (1 / 3))

        x = 0.5 + r * math.sin(theta) * math.cos(phi)
        y = 0 + r * math.sin(theta) * math.sin(phi)
        z = 0.9 + r * math.cos(theta)
        self.target = np.array([x, y, z])
        print('Set target as', self.target)

        # drawing a visual indicator of the target
        # currently works by exploiting a bug within pyBullet, extremly unsmart but can be used for testing 
        p.resetBasePositionAndOrientation(bodyUniqueId=self.target_body, posObj=self.target,
                                          ornObj=np.array([0, 0, 0, 0]))
        self.target_body = p.createMultiBody(baseMass=0, baseVisualShapeIndex=self.targetSphere,
                                             basePosition=self.target)

        if not self.obs:
            return

        for i in range(0, self.NUM_OBS):
            # randomly select a point in the spere around Jaco
            # set it as the obstacle
            phi = random() * 2 * math.pi
            cosTheta = random() * 2 - 1
            u = random()

            theta = math.acos(cosTheta)
            r = 0.3 * (u ** (1 / 3))

            x = 0.5 + r * math.sin(theta) * math.cos(phi)
            y = 0 + r * math.sin(theta) * math.sin(phi)
            z = 0.9 + r * math.cos(theta)
            self.obsTar[i] = np.array([x, y, z])

            while (self.overlapping(i)):
                chosen = choice([True, False])
                index = randint(1, 3)
                inc = -0.1
                if (chosen):
                    inc = 0.1
                if (index == 1):
                    x += inc
                elif (index == 2):
                    y += inc
                else:
                    z += inc
                self.obsTar[i] = np.array([x, y, z])

            p.resetBasePositionAndOrientation(bodyUniqueId=self.obs_body[i], posObj=self.obsTar[i],
                                              ornObj=np.array([0, 0, 0, 0]))
            self.obs_body[i] = p.createMultiBody(baseMass=0, baseVisualShapeIndex=self.obsSphereVis,
                                                 baseCollisionShapeIndex=self.obsSphereCol, basePosition=self.obsTar[i])

    def overlapping(self, i):
        distance_vector = self.obsTar[i] - self.target
        if (np.linalg.norm(distance_vector) < 0.25):
            return True
        for j in range(0, self.NUM_OBS):
            if (i == j):
                continue
            else:
                distance_vector = self.obsTar[i] - self.obsTar[j]
                if (np.linalg.norm(distance_vector) < 0.3):
                    return True
        for j in range(0, 9):
            observation = self.jaco.get_link_pose(j)[:3]
            distance = np.linalg.norm(observation - self.obsTar[i])
            if (distance < 0.3):
                return True
        return False

    def reset(self):
        # getting Jaco back to his starting position
        startingPosition = [0, 3.14, 3.14 / 2, -3.14 / 2, -3.14 / 2, 0, 0, 0, 0]  # set joint positiodonen goal
        self.jaco._hard_set_joint_positions(startingPosition)

        # simulate a zero-action and update Jaco
        action = np.zeros((6))

        # self.jaco.set_end_effector_velocity(action, [0, 1, 2, 3, 4, 5])
        self.jaco.set_joint_velocity_goal(np.concatenate((action, np.zeros(self.jaco._num_jnt_gripper))))
        self.jaco.close_gripper()
        self.jaco.update()
        time.sleep(0.1)

        self.seed()
        observation = self.get_observation()
        self.numberOfSteps = 0
        # return the initial observation
        if (self.obs):
            return np.concatenate((self.target, observation[:3], *self.obsTar, *self.get_links()), 0)
        else:
            return np.concatenate((self.target, observation[:3]), 0)

    def render(self, mode='human', close=False):
        pass
