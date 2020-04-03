from .FlightModel_2 import FlightModel

import numpy as np
import gym
from gym import spaces


class PlaneEnvironment(gym.Env):
    """
    Plane Environment for Open AI Gym
    Based on FlightModel for basic 2D plane simulation
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(PlaneEnvironment, self).__init__()

        self.current_step = 0
        self.FlightModel = FlightModel()

        # ACTIONS
        # Possible thrust values
        self.action_space = spaces.Discrete(10)

        # STATES
        self.observation_space = spaces.Box(
            np.array([0, 0, -100, -1000]), np.array([1000000, 100000, 100, 1000])
        )

    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_step = 0
        self.FlightModel = FlightModel()
        obs = FlightModel().obs
        return obs

    def step(self, action):
        # Execute one time step within the environment
        self.current_step += 1
        obs, reward, done = self.FlightModel.compute_episode(action)
        if self.current_step > 1000:
            done = True
        return obs, reward, done, {}

    def render(self, mode="human", close=False):
        print(self.current_step, self.FlightModel.Pos)
