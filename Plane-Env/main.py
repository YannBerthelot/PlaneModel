import os
import time
import numpy as np
from env.FlightModel import FlightModel
from env.FlightEnv import PlaneEnvironment
from utils import write_to_txt
from env.AnimatePlane import animate_plane
from tensorforce import Agent, Environment
from tensorforce.core.networks import AutoNetwork
from env.graph_utils import plot_duo, plot_multiple
import pandas as pd
import matplotlib.pyplot as plt

def train_info(i, n_episodes, start_time):
    temp_time = time.time() - start_time
    time_per_episode = temp_time / (i + 1)
    print(
        "episode : ",
        i,
        "/",
        n_episodes,
        " time per episode",
        round(time_per_episode, 2),
        "seconds. ",
        "estimated time to finish",
        int((time_per_episode * n_episodes) - temp_time),
        "seconds.",
    )


def terminal_info(thrust_vec, theta_vec, episode_length, states, actions, sum_rewards):
    print(FlightModel.action_vec[actions])
    print(states)
    print(
        "mean reward",
        int(sum_rewards / episode_length),
        "mean action",
        round(np.mean(thrust_vec), 2),
        round(np.mean(theta_vec), 2),
        "std",
        round(np.std(thrust_vec), 2),
        round(np.std(theta_vec), 2),
    )
    print("episode length", episode_length)


def train(n_episodes, max_step_per_episode):
    """
    Train agent for n_episodes
    """
    start_time = time.time()
    environment.FlightModel.max_step_per_episode = max_step_per_episode
    global_reward_vec = []
    global_reward_mean_vec = []
    for i in range(n_episodes):
        sum_rewards = 0.0
        episode_length = 0
        train_info(i, n_episodes, start_time)
        # Initialize episode
        states = environment.reset()
        internals = agent.initial_internals()
        terminal = False
        thrust_vec = []
        theta_vec = []
        reward_vec = []
        while not terminal:
            episode_length += 1
            # Episode timestep
            actions = agent.act(states=states)
            thrust_vec.append(round(FlightModel.action_vec[actions][0], 2))
            theta_vec.append(round(FlightModel.action_vec[actions][1], 2))
            states, terminal, reward = environment.execute(actions=actions)
            reward_vec.append(reward)
            agent.observe(terminal=terminal, reward=reward)
            sum_rewards += reward
            
            if terminal:
                terminal_info(
                    thrust_vec, theta_vec, episode_length, states, actions, sum_rewards
                )
        global_reward_vec.append(sum_rewards)
        global_reward_mean_vec.append(np.mean(global_reward_vec))
    show_policy(thrust_vec, theta_vec, reward_vec)
    end_time = time.time()
    total_time = end_time - start_time
    print("total_time", total_time, "total reward", np.sum(reward_vec))
    Series = [global_reward_vec,global_reward_mean_vec]
    labels = ["Reward","Mean reward"]
    xlabel = "time (s)"
    ylabel = "Reward"
    title = "Global Reward vs time"
    plot_multiple(Series, labels, xlabel, ylabel, title, save_fig=True, path="env")

def test(n_episodes, max_step_per_episode):
    start_time = time.time()
    environment.FlightModel.max_step_per_episode = max_step_per_episode
    sum_rewards = 0.0
    for _ in range(n_episodes):
        states = environment.reset()
        internals = agent.initial_internals()
        terminal = False
        episode_length = 0
        thrust_vec = []
        theta_vec = []
        reward_vec = []
        while not terminal:
            episode_length += 1
            actions, internals = agent.act(
                states=states, internals=internals, evaluation=True
            )
            thrust_vec.append(round(FlightModel.action_vec[actions][0], 2))
            theta_vec.append(round(FlightModel.action_vec[actions][1], 2))
            states, terminal, reward = environment.execute(actions=actions)
            reward_vec.append(reward)
            sum_rewards += reward
            if terminal:
                terminal_info(
                    thrust_vec, theta_vec, episode_length, states, actions, sum_rewards
                )
    show_policy(thrust_vec, theta_vec, reward_vec)
    end_time = time.time()
    total_time = end_time - start_time
    print("total_time", total_time, "total reward", np.sum(reward_vec))
    print("Mean episode reward:", sum_rewards / n_episodes)
    return sum_rewards / n_episodes


def show_policy(thrust_vec, theta_vec, reward_vec):
    Series = [thrust_vec, theta_vec]
    labels = ["Thrust", "Theta"]
    xlabel = "time (s)"
    ylabel = "Force intensity (N)/Angle value (°)"
    title = "Policy vs time"
    plot_duo(Series, labels, xlabel, ylabel, title, save_fig=True, path="env")
    cumulative_reward = np.cumsum(reward_vec)
    Series = [reward_vec, cumulative_reward]
    labels = ["Reward", "Cumulative reward"]
    xlabel = "time (s)"
    ylabel = "Reward"
    title = "Reward vs time"
    plot_multiple(Series, labels, xlabel, ylabel, title, save_fig=True, path="env")
    


# Instantiate our Flight Model
FlightModel = FlightModel()
# Instantiane our environment
environment = PlaneEnvironment()
# Instantiate a Tensorforce agent
policy = dict(network=dict(type="auto", size=128, depth=6))

# agent = Agent.create(
#     agent="tensorforce",
#     environment=environment,  # alternatively: states, actions, (max_episode_timesteps)
#     memory=1000,
#     update=dict(unit="episodes", batch_size=1),
#     optimizer=dict(type="adam", learning_rate=1e-3),
#     policy=policy,
#     objective="value",
#     reward_estimation=dict(horizon=1000, estimate_actions=True),
#     exploration=0.02,
#     parallel_interactions=4,
#     seed=124,
# )
mean_reward_vec = []
for i in range(25,26):
    agent = Agent.create(
        agent="dueling_dqn",
        environment=environment,  # alternatively: states, actions, (max_episode_timesteps)
        memory=10000,
        horizon = 25,
        exploration=0.05,
        batch_size=32,
        discount=0.1,
        seed=124,
        l2_regularization =0.001,
        estimate_terminal = True,
    )



    # Define train parameters
    max_step_per_episode = 1000
    n_episodes = 2000
    # Train agent
    train(n_episodes, max_step_per_episode)

    # Define test parameters
    n_episodes = 10
    # # Test Agent
    mean_reward_vec.append(test(n_episodes, max_step_per_episode))
    environment.FlightModel.plot_graphs(save_figs=True,path="env")

print(mean_reward_vec)
pd.Series(mean_reward_vec).to_csv('Discount.csv')
# Save last run positions
write_to_txt(environment)
# Animate last run positions
# animate_plane()


# End
agent.close()
environment.close()
