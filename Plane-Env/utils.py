import os
import itertools
from tensorforce import Agent, Environment
import time
from env.graph_utils import plot_duo, plot_multiple
import numpy as np
import matplotlib.pyplot as plt
from env.FlightEnv import PlaneEnvironment


def write_to_txt(environment):
    cur_path = os.path.dirname(__file__)
    new_path = os.path.relpath("env/positions.txt", cur_path)
    positions = environment.FlightModel.Pos_vec
    text_file = open(new_path, "w")
    n = text_file.write(str(positions))
    text_file.close()
    angles = environment.FlightModel.theta_vec
    text_file = open("env/angles.txt", "w")
    n = text_file.write(str(angles))
    text_file.close()


def write_results_to_txt(param_dict):
    cur_path = os.path.dirname(__file__)
    new_path = os.path.relpath("env/results.txt", cur_path)
    text_file = open(new_path, "w")
    n = text_file.write(str(param_dict))
    text_file.close()


def GridSearchTensorForce(
    environment, param_grid_list, max_step_per_episode, n_episodes
):
    lists = param_grid_list.values()
    param_combinations = list(itertools.product(*lists))

    total_param_combinations = len(param_combinations)
    print("Number of combinations", total_param_combinations)
    scores = []
    names = []
    for i, params in enumerate(param_combinations, 1):
        print("Combination", i, "/", total_param_combinations)
        # fill param dict with params
        param_grid = {}
        for param_index, param_name in enumerate(param_grid_list):
            param_grid[param_name] = params[param_index]

        agent = Agent.create(
            agent=param_grid["agent"],
            environment=PlaneEnvironment(),  # alternatively: states, actions, (max_episode_timesteps)
            memory=param_grid["memory"],
            network=dict(
                type=param_grid["network"],
                size=param_grid["size"],
                depth=param_grid["depth"],
            ),
            horizon=param_grid["horizon"],
            exploration=param_grid["exploration"],
            batch_size=param_grid["batch_size"],
            discount=param_grid["discount"],
            seed=param_grid["seed"],
            estimate_terminal=param_grid["estimate_terminals"],
        )
        scores.append(trainer(environment, agent, max_step_per_episode, n_episodes))
        names.append(str(param_grid))
    dict_scores = dict(zip(names, scores))
    write_results_to_txt(dict_scores)
    best_model = min(dict_scores, key=dict_scores.get)
    print("best model", best_model)
    print("best model score", dict_scores[best_model])


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
    plt.close(fig="all")


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
    print("actions", actions, "states", states)
    print(
        "mean reward",
        int(sum_rewards / episode_length),
        "mean action",
        round(np.mean(thrust_vec), 2),
        round(np.mean(theta_vec), 2),
        "std",
        round(np.std(thrust_vec), 2),
        round(np.std(theta_vec), 2),
        "episode length",
        episode_length,
    )


def train(environment, agent, n_episodes, max_step_per_episode):
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
            # print("actions", actions)
            thrust_vec.append(round(actions["thrust"], 2))
            theta_vec.append(round(actions["theta"], 2))
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
    # show_policy(thrust_vec, theta_vec, reward_vec)
    end_time = time.time()
    total_time = end_time - start_time
    print("total_time", total_time, "total reward", np.sum(reward_vec))
    Series = [global_reward_vec, global_reward_mean_vec]
    labels = ["Reward", "Mean reward"]
    xlabel = "time (s)"
    ylabel = "Reward"
    title = "Global Reward vs time"
    plot_multiple(Series, labels, xlabel, ylabel, title, save_fig=True, path="env")


def test(environment, agent, n_episodes, max_step_per_episode):
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
            thrust_vec.append(round(actions["thrust"], 2))
            theta_vec.append(round(actions["theta"], 2))
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


def trainer(environment, agent, max_step_per_episode, n_episodes, n_episodes_test=1):
    # Train agent
    train(environment, agent, n_episodes, max_step_per_episode)
    # # Test Agent
    test(environment, agent, n_episodes_test, max_step_per_episode)
    # environment.FlightModel.plot_graphs(save_figs=True, path="env")
    agent.close()
    environment.close()
    return environment.FlightModel.Pos[0]
