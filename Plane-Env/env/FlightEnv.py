import numpy as np
from tensorforce.environments import Environment
from .FlightModel import FlightModel


class PlaneEnvironment(Environment):
    def __init__(self, task="take-off", max_step_per_episode=1000):
        print("task", task)
        super().__init__()
        self.task = task
        self.FlightModel = FlightModel(task=self.task)
        self.NUM_ACTIONS = len(self.FlightModel.action_vec)
        self.NUM_THRUST = len(self.FlightModel.thrust_act_vec)
        self.NUM_THETA = len(self.FlightModel.theta_act_vec)
        self.max_step_per_episode = max_step_per_episode
        self.finished = False
        self.episode_end = False
        self.STATES_SIZE = len(self.FlightModel.obs)
        self.overtime = False
        self.counter = 0

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
        self.FlightModel = FlightModel(task=self.task)
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
        self.overtime = self.FlightModel.timestep > self.max_step_per_episode
        if self.task == "take-off":
            # Episode success/Take-off altitude reached
            self.finished = self.FlightModel.Pos[1] > 25
            # Runway end reached
            max_dist_reached = self.FlightModel.Pos[0] > 5000
            # Episode failure
            self.episode_end = self.overtime or max_dist_reached
            return self.finished or self.episode_end
        elif self.task == "level-flight":
            # Episode end (no notion of success or failure here)
            self.episode_end = self.overtime
            return self.episode_end

    def reward(self):
        self.counter += 1
        if self.task == "take-off":
            if self.finished:
                reward = np.log(((5000 - self.FlightModel.Pos[0]) ** 2))
            else:
                reward = -1
            return reward
        elif self.task == "level-flight":

            initial_alt = self.FlightModel.initial_altitude
            alt_diff = self.FlightModel.Pos[1] - initial_alt
            v = self.FlightModel.V[1]
            reward = -alt_diff / (abs(v) + 1)
            reward = -np.sqrt(abs(reward))

            if (self.counter % 1000 == 0) & (self.counter > 0):
                print(
                    "reward",
                    round(reward, 0),
                    "inital_alt",
                    initial_alt,
                    "alt",
                    round(self.FlightModel.Pos[1], 0),
                )
            return reward

