import numpy as np
from tensorforce.environments import Environment
from .FlightModel import FlightModel


class PlaneEnvironment(Environment):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.FlightModel = FlightModel(self.mode)
        self.NUM_ACTIONS = len(self.FlightModel.action_vec)
        self.NUM_THRUST = len(self.FlightModel.thrust_act_vec)
        self.NUM_THETA = len(self.FlightModel.theta_act_vec)
        self.max_step_per_episode = 30000
        self.finished = False
        self.episode_end = False
        self.STATES_SIZE = len(self.FlightModel.obs)
        self.initial_altitude = self.FlightModel.Pos[1]
        self.reward_value = 0
        self.landed = False
        self.overspeed = False
        self.over_z = False
        self.actions_store = None

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
        self.FlightModel = FlightModel(self.mode)
        self.initial_altitude = self.FlightModel.Pos[1]
        state = self.FlightModel.obs
        self.episode_end = False
        return state

    def execute(self, actions):
        # print("actions", actions)
        self.actions_store = actions
        next_state = self.FlightModel.compute_timestep(actions)
        terminal = self.terminal()
        reward = self.reward()
        return next_state, terminal, reward

    def terminal(self):
        time = self.FlightModel.timestep > self.max_step_per_episode
        if time:
            print("time")
        if abs(self.FlightModel.V[1]) > 20:
            self.overspeed = True
        else:
            self.overspeed = False
        self.episode_end = time or (self.FlightModel.crashed) or (self.landed)
        if self.FlightModel.crashed:
            print("crash")
        self.landed = (
            (abs(self.FlightModel.Pos[1]) < 10)
            and (abs(self.FlightModel.V[0]) < 1)
            and not (self.FlightModel.crashed)
        )
        if self.episode_end:
            self.FlightModel.plot_graphs(save_figs=True)
            print(
                "initial altitude",
                self.initial_altitude,
                "final altitude",
                self.FlightModel.Pos[1],
                "reward",
                self.reward(),
            )
        return self.episode_end

    def reward(self):
        V = abs(self.FlightModel.V[0]) / 100
        z = abs(self.FlightModel.Pos[1]) / 3000
        if not (self.FlightModel.crashed):

            # self.reward_value = (
            #     -((1 + max(1 * V / 100, 0.0)) ** (10 * z / 3000)) / 1000
            # )
            self.reward_value = -(3 * V + 2 * z + V * z)
            if self.actions_store["thrust"] > 2:
                self.reward_value += 5

            if self.landed:
                print("landed")
                self.reward_value += 1000
        else:
            self.reward_value = -10

        return self.reward_value
