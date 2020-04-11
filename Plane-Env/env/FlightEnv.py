from .FlightModel import FlightModel
import numpy as np
from math import ceil
from tensorforce.environments import Environment


class PlaneEnvironment(Environment):
    def __init__(self):
        super().__init__()
        self.FlightModel = FlightModel()
        self.NUM_ACTIONS = len(self.FlightModel.action_vec)
        self.max_step_per_episode = 1000

    def states(self):
        return dict(type="float", shape=(4,))

    def actions(self):
        return dict(type="int", num_values=self.NUM_ACTIONS)

    # Optional, should only be defined if environment has a natural maximum
    # episode length
    def max_episode_timesteps(self):
        return self.max_step_per_episode

    # Optional
    def close(self):
        super().close()

    def reset(self):
        state = np.zeros(shape=(4,))
        self.FlightModel.reset()
        return state

    def execute(self, actions):
        assert 0 <= actions.item() <= self.NUM_ACTIONS
        next_state = self.FlightModel.compute_timestep(actions)
        terminal = self.terminal()
        reward = self.reward(terminal)
        return next_state, terminal, reward

    def terminal(self):
        self.finished = self.FlightModel.Pos[1] > 1000
        self.episode_end = self.FlightModel.timestep > self.max_step_per_episode
        return self.finished or self.episode_end

    def reward(self, terminal):
        reward = (self.FlightModel.Pos[1] ** 2) - (self.FlightModel.Pos[0] / 10)
        # if terminal:
        #     if self.FlightModel.finished:
        #         reward += 1000
        #     else:
        #         reward += -1000
        # else:
        #     if self.FlightModel.Pos[1] > 0:
        #         reward += (
        #             np.mean(self.FlightModel.V_vec[0])
        #             + ceil(np.mean(self.FlightModel.V_vec[1])) ** 2
        #         )
        #     else:
        #         reward += np.mean(self.FlightModel.V_vec[0])
        return reward
