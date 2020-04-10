from .FlightModel import FlightModel

import numpy as np
import gym
from gym import spaces

class PlaneEnvironment_2(gym.Env):
    """
    Plane Environment for Open AI Gym
    Based on FlightModel for basic 2D plane simulation
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(PlaneEnvironment, self).__init__()

        self.current_step = 0
        self.FlightModel = FlightModel()
        self.NUM_ACTIONS = len(self.FlightModel.action_vec)
        self.obs = []

        # ACTIONS
        # Possible thrust values
        self.action_space = spaces.Discrete(self.NUM_ACTIONS)
        self.action = 0

        # STATES
        self.observation_space = spaces.Box(
            np.array([0, 0, -1000, -1000]), np.array([100000, 10000, 1000, 1000])
        )

    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_step = 0
        self.FlightModel = FlightModel()
        obs = FlightModel().obs
        return obs

    def step(self, action):
        # Execute one time step within the environment
        self.action = action
        self.current_step += 1
        obs, reward, done = self.FlightModel.compute_episode(action)
        self.obs = obs
        # if self.current_step > 1000:
        #     done = True
        return obs, reward, done, {}

    def render(self, mode="human", close=False):
        print(
            "step:",
            self.current_step,
            " Pos ",
            self.FlightModel.Pos,
            "Speed",
            self.FlightModel.V,
            "OBS",
            self.obs,
            "Action",
            self.FlightModel.action_vec[self.action],
        )
        # self.FlightModel._animate_plane()

)

