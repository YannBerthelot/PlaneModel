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
        self.initial_altitude = self.FlightModel.Pos[1]
        self.reward_value = 0

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
        self.FlightModel = FlightModel()
        self.initial_altitude = self.FlightModel.Pos[1]
        state = self.FlightModel.obs
        self.episode_end = False
        return state

    def execute(self, actions):
        next_state = self.FlightModel.compute_timestep(actions)
        terminal = self.terminal()
        reward = self.reward()
        return next_state, terminal, reward

    def terminal(self):
        self.episode_end = (self.FlightModel.timestep > self.max_step_per_episode) or (
            self.FlightModel.crashed
        )
        if self.episode_end:
            self.FlightModel.plot_graphs(save_figs=True)
            print(
                "initial altitude",
                self.initial_altitude,
                "final altitude",
                self.FlightModel.Pos[1],
                "reward",
                self.reward_value,
            )
        return self.episode_end

    def reward(self):
        self.reward_value = -abs(self.initial_altitude - self.FlightModel.Pos[1]) / 10
        if self.FlightModel.crashed:
            self.reward_value = -1000
        return self.reward_value
