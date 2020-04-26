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
param_grid_list = {}
param_grid_list["dueling_dqn"] = {
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

param_grid_list["PPO"] = {
    "batch_size": [10],
    "update_frequency": [10],
    "learning_rate": [1e-3],
    "subsampling_fraction": [0.2],
    "optimization_steps": [10],
    "likelihood_ratio_clipping": [0.1],
    "discount": [0.99],
    "estimate_terminal": [False],
    "multi_step": [10],
    "learning_rate_critic": [1e-3],
    "exploration": [0.01],
    "variable_noise": [0.0],
    "l2_regularization": [0.0001, 0.001, 0],
    "entropy_regularization": [0.001, 0.0001, 0.01],
}

max_step_per_episode = 100
n_episodes = 5000

GridSearchTensorForce(
    environment, param_grid_list["PPO"], max_step_per_episode, n_episodes
)

# Save last run positions
write_to_txt(environment)
# Animate last run positions
# animate_plane()

