import os

import numpy as np
from env.FlightModel import FlightModel
from env.FlightEnv import PlaneEnvironment
from utils import (
    write_to_txt,
    GridSearchTensorForce,
    show_policy,
    train_info,
    terminal_info,
    train,
    test,
)
from env.AnimatePlane import animate_plane
from tensorforce import Agent, Environment
from tensorforce.core.networks import AutoNetwork
import pandas as pd
import matplotlib.pyplot as plt

# Instantiate our Flight Model
FlightModel = FlightModel()
# Instantiane our environment
environment = PlaneEnvironment()
# Instantiate a Tensorforce agent

param_grid_list = {
    "horizon": [10],
    "agent": ["dueling_dqn"],
    "memory": [2000],
    "network": ["auto"],
    "size": [64],
    "depth": [10],
    "exploration": [0.02],
    "batch_size": [32],
    "discount": [0.1],
    "seed": [124],
    "estimate_terminals": [True],
}

max_step_per_episode = 100
n_episodes = 150000

GridSearchTensorForce(environment, param_grid_list, max_step_per_episode, n_episodes)

# Save last run positions
write_to_txt(environment)
# Animate last run positions
# animate_plane()

