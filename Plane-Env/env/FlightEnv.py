from .FlightModel import FlightModel
import numpy as np
from tensorforce.environments import Environment


class PlaneEnvironment(Environment):
    def __init__(self):
        super().__init__()
        self.FlightModel = FlightModel()
        self.NUM_ACTIONS = len(self.FlightModel.action_vec)

    def states(self):
        return dict(type="float", shape=(4,))

    def actions(self):
        return dict(type="int", num_values=self.NUM_ACTIONS)

    # Optional, should only be defined if environment has a natural maximum
    # episode length
    def max_episode_timesteps(self):
        return 1000

    # Optional
    def close(self):
        super().close()

    def reset(self):
        self.FlightModel.timestep = 0
        state = np.zeros(shape=(4,))
        return state

    def execute(self, actions):
        assert 0 <= actions.item() <= self.NUM_ACTIONS
        next_state, terminal = self.FlightModel.compute_episode(actions)
        reward = self.reward(terminal)
        return next_state, terminal, reward

    def reward(self, terminal):
        reward = 0
        if terminal:
            if self.FlightModel.finished:
                reward += 1000
            else:
                reward += -1000
        else:
            if self.FlightModel.Pos[1] > 0:
                reward += self.FlightModel.Pos[0] + self.FlightModel.Pos[1]
            else:
                reward += self.FlightModel.Pos[0]
        return reward
