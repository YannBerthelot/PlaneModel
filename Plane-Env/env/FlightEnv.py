import numpy as np
from tensorforce.environments import Environment
from .FlightModel import FlightModel


class PlaneEnvironment(Environment):
    def __init__(self):
        super().__init__()
        self.FlightModel = FlightModel()
        self.NUM_ACTIONS = len(self.FlightModel.action_vec)
        self.NUM_THRUST = len(self.FlightModel.thrust_act_vec)
        self.NUM_THETA = len(self.FlightModel.theta_act_vec)
        self.max_step_per_episode = 1000
        self.finished = False
        self.episode_end = False
        self.STATES_SIZE = len(self.FlightModel.obs)

    def states(self):
        return dict(type="float", shape=(self.STATES_SIZE,))

    def actions(self):
        return {
            "thrust": dict(type="int", num_values=self.NUM_THRUST),
            "theta": dict(type="int", num_values=self.NUM_THETA),
        }

    # Optional, should only be defined if environment has a natural maximum
    # episode length
    def max_episode_timesteps(self):
        return self.max_step_per_episode

    # Optional
    def close(self):
        super().close()

    def reset(self):
        state = np.zeros(shape=(self.STATES_SIZE,))
        self.FlightModel = FlightModel()
        return state

    def execute(self, actions):
        reward = 0
        nb_timesteps = 1
        for i in range(1, nb_timesteps + 1):
            next_state = self.FlightModel.compute_timestep(actions, nb_timesteps)
            reward += self.reward()
            if self.terminal():
                reward = reward / i
                break
        if i == nb_timesteps:
            reward = reward / nb_timesteps
        # reward = self.reward()
        return next_state, self.terminal(), reward

    def terminal(self):
        self.finished = self.FlightModel.Pos[1] > 25
        self.episode_end = (self.FlightModel.timestep > self.max_step_per_episode) or (
            self.FlightModel.Pos[0] > 5000
        )
        return self.finished or self.episode_end

    def reward(self):
        if self.finished:
            reward = np.log(((5000 - self.FlightModel.Pos[0]) ** 2))
        else:
            reward = -1
        return reward
