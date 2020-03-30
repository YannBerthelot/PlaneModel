import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np

MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_SHARE_PRICE = 5000
MAX_OPEN_POSITIONS = 5
MAX_STEPS = 20000

INITIAL_ACCOUNT_BALANCE = 10000


class PlaneEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""

    metadata = {"render.modes": ["human"]}

    def __init__(self, df):
        super(PlaneEnv, self).__init__()
        self.delta_t = 1
        self.len_experiment = 1000

        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.action_space = spaces.Box(low=0, high=THRUST_MAX)

        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.Tuple(
            (spaces.Box(0, 20000, shape=(1, 1)), spaces.Box(0, 1000, shape=(1, 1)))
        )

    def _next_observation(self):

        # Append additional data and scale each value to between 0-1
        obs = np.append(frame, [[a, b, c]], axis=0)

        return obs

    def _take_action(self, action):
        # Set the current price to a random price within the time step

        if action_type < 1:
            # Buy amount % of balance in shares
            total_possible = int(self.balance / current_price)
            shares_bought = int(total_possible * amount)
            prev_cost = self.cost_basis * self.shares_held
            additional_cost = shares_bought * current_price

            self.balance -= additional_cost
            self.cost_basis = (prev_cost + additional_cost) / (
                self.shares_held + shares_bought
            )
            self.shares_held += shares_bought

        elif action_type < 2:
            # Sell amount % of shares held
            shares_sold = int(self.shares_held * amount)
            self.balance += shares_sold * current_price
            self.shares_held -= shares_sold
            self.total_shares_sold += shares_sold
            self.total_sales_value += shares_sold * current_price

        self.net_worth = self.balance + self.shares_held * current_price

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        if self.shares_held == 0:
            self.cost_basis = 0

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)

        self.current_step += 1

        reward = ####
        done = ######

        obs = self._next_observation()

        return obs, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.alpha = 10
        self.Acceleration = [0,0]
        self.V = [225,0]
        self.Pos = [0,10000]
        self.dyna_vec = [self.Acceleration,self.V,self.Pos]
        self.Altitude_factor = 1
        self.A_vec_x = []
        self.A_vec_y = []
        self.V_vec_x = []
        self.V_vec_y = []
        self.Pos_vec_x = []
        self.Pos_vec_y = []
        self.Force_vec_y = []
        self.Force_vec_x = []
        self.Lift_vec = []
        self.P_vec = []
        self.T_vec = []
        self.Drag_vec = []
        self.T_vec_x = []
        self.Drag_vec_x = []
        self.current_step = 0
        return self._next_observation()

    def render(self, mode="human", close=False):
        # Render the environment to the screen

        print(f"Step: {self.current_step}")
        print(f"Balance: {self.balance}")
        print(f"Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})")
        print(
            f"Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})"
        )
        print(f"Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})")
        print(f"Profit: {profit}")
