import os
import time
import itertools
import shutil
from collections import namedtuple
import numpy as np
from tensorforce import Agent
from env.graph_utils import plot_duo, plot_multiple


def write_to_txt_general(data, path):
    cur_path = os.path.dirname(__file__)
    new_path = os.path.relpath("env/" + str(path), cur_path)
    text_file = open(new_path, "w")
    n = text_file.write(str(data))
    text_file.close()


def write_pos_and_angles_to_txt(environment, path):
    write_to_txt_general(environment.FlightModel.Pos_vec, path + "/positions.txt")
    write_to_txt_general(environment.FlightModel.theta_vec, path + "/angles.txt")


def write_combination_to_txt(param_dict, folder=None):
    cur_path = os.path.dirname(__file__)
    if folder:
        new_path = os.path.join("env", "Graphs", str(folder), "params.txt")
    else:
        new_path = os.path.relpath("env/params.txt", cur_path)

    text_file = open(new_path, "w")
    n = text_file.write(str(param_dict))
    text_file.close()


def create_agent(param_grid, i, directory, environment):
    return Agent.create(
        agent="ppo",
        environment=environment,
        # Automatically configured network
        network=dict(
            type="auto",
            size=64,
            depth=8
            final_size=1,
        ),
        # Optimization
        batch_size=param_grid["batch_size"],
        update_frequency=param_grid["update_frequency"],
        learning_rate=param_grid["learning_rate"],
        subsampling_fraction=param_grid["subsampling_fraction"],
        optimization_steps=param_grid["optimization_steps"],
        # Reward estimation
        likelihood_ratio_clipping=param_grid["likelihood_ratio_clipping"],
        discount=param_grid["discount"],
        estimate_terminal=param_grid["estimate_terminal"],
        # Critic
        critic_network="auto",
        critic_optimizer=dict(
            optimizer="adam",
            multi_step=param_grid["multi_step"],
            learning_rate=param_grid["learning_rate_critic"],
        ),
        # Preprocessing
        preprocessing=None,
        # Exploration
        exploration=param_grid["exploration"],
        variable_noise=param_grid["variable_noise"],
        # Regularization
        l2_regularization=param_grid["l2_regularization"],
        entropy_regularization=param_grid["entropy_regularization"],
        # TensorFlow etc
        name="agent_" + str(i),
        device=None,
        parallel_interactions=5,
        seed=124,
        execution=None,
        recorder=dict(directory=directory, frequency=1000),
        summarizer=None,
        saver=dict(directory=directory, filename="agent_" + str(i)),
    )


def gridsearch_tensorforce(
    environment, param_grid_list, max_step_per_episode, n_episodes
):
    GridSearch = namedtuple("GridSearch", ["scores", "names"])
    gridsearch = GridSearch([], [])

    # Compute the different parameters combinations
    param_combinations = list(itertools.product(*param_grid_list.values()))
    for i, params in enumerate(param_combinations, 1):
        if not os.path.exists(os.path.join("env", "Graphs", str(i))):
            os.mkdir(os.path.join("env", "Graphs", str(i)))
        # fill param dict with params
        param_grid = {
            param_name: params[param_index]
            for param_index, param_name in enumerate(param_grid_list)
        }
        directory = os.path.join(os.getcwd(), "env", "Models", str(i))
        if os.path.exists(directory):
            shutil.rmtree(directory, ignore_errors=True)

        agent = create_agent(param_grid, i, directory, environment)
        # agent = Agent.load(directory="data/checkpoints")
        gridsearch.scores.append(
            trainer(
                environment,
                agent,
                max_step_per_episode,
                n_episodes,
                combination=i,
                total_combination=len(param_combinations),
            )
        )
        store_results_and_graphs(i, environment, param_grid)
        gridsearch.names.append(str(param_grid))
    dict_scores = dict(zip(gridsearch.names, gridsearch.scores))
    write_to_txt_general(dict_scores, "results.txt")
    best_model = min(dict_scores, key=dict_scores.get)
    print(
        "best model",
        best_model,
        "number",
        np.argmin(gridsearch.scores),
        "score",
        dict_scores[best_model],
    )


def store_results_and_graphs(i, environment, param_grid):
    write_pos_and_angles_to_txt(environment, "")
    write_combination_to_txt(param_grid, folder=str(i))


def show_policy(
    thrust_vec, theta_vec, x, z, distances, combination, title="Policy vs time"
):
    plot_duo(
        Series=[thrust_vec, theta_vec],
        labels=["Thrust", "Theta"],
        xlabel="time (s)",
        ylabel="Force intensity (N)/Angle value (°)",
        title=title,
        save_fig=True,
        path="env",
        folder=str(combination),
        time=True,
    )
    plot_duo(
        Series=[x, z],
        labels=["x", "z"],
        xlabel="time (s)",
        ylabel="Distance (m)",
        title=title+"_pos",
        save_fig=True,
        path="env",
        folder=str(combination),
        time=True,
    )

    plot_multiple(
        Series=[distances],
        labels=["TO-Distance"],
        xlabel="episodes",
        ylabel="TO-Distance (m)",
        title="Distance vs episodes",
        save_fig=True,
        path="env",
        folder=str(combination),
        time=False,
    )


def train_info(i, n_episodes, start_time, combination):
    temp_time = time.time() - start_time
    time_per_episode = temp_time / (i + 1)
    print(
        "combination : ",
        combination,
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


def terminal_info(episode, states, actions):
    print("actions", actions, "states", states)
    print(
        "mean reward",
        np.mean(episode.rewards),
        "mean action",
        round(np.mean(episode.thrust_values), 2),
        round(np.mean(episode.theta_values), 2),
        "std",
        round(np.std(episode.thrust_values), 2),
        round(np.std(episode.theta_values), 2),
        "episode length",
        len(episode.rewards),
    )


def run(
    environment,
    agent,
    n_episodes,
    max_step_per_episode,
    combination,
    total_combination,
    batch,
    test=False,
):
    """
    Train agent for n_episodes
    """
    environment.FlightModel.max_step_per_episode = max_step_per_episode
    Score = namedtuple("Score", ["reward", "reward_mean", "distance"])
    score = Score([], [], [])

    start_time = time.time()
    for i in range(1, n_episodes + 1):
        # Variables initialization
        Episode = namedtuple(
            "Episode",
            ["rewards", "thrust_values", "theta_values", "x_values", "z_values"],
        )
        episode = Episode([], [], [], [], [])

        if total_combination == 1 and (
            i % 50 == 0
        ):  # Print training information every 50 episodes
            train_info(i, n_episodes, start_time, combination)

        # Initialize episode
        states = environment.reset()
        internals = agent.initial_internals()
        terminal = False

        while not terminal:  # While an episode has not yet terminated

            if test:  # Test mode (deterministic, no exploration)
                actions, internals = agent.act(
                    states=states, internals=internals, evaluation=True
                )
                states, terminal, reward = environment.execute(actions=actions)
            else:  # Train mode (exploration and randomness)
                actions = agent.act(states=states)
                states, terminal, reward = environment.execute(actions=actions)
                agent.observe(terminal=terminal, reward=reward)

            episode.thrust_values.append(round(actions["thrust"], 2))
            episode.theta_values.append(round(actions["theta"], 2))
            episode.x_values.append(environment.FlightModel.Pos[0])
            episode.z_values.append(environment.FlightModel.Pos[1])
            episode.rewards.append(reward)
            # if terminal and (i % 100 == 0):
            #     terminal_info(
            #         episode, states, actions,
            #     )
        score.reward.append(np.sum(episode.rewards))
        score.reward_mean.append(np.mean(score.reward))
        task = "level-flight"
        if task == "take-off":
            score.distance.append(environment.FlightModel.Pos[0])
        elif task == "level-flight":
            initial_alt = environment.FlightModel.initial_altitude
            diff_alt = -abs(initial_alt - environment.FlightModel.Pos[1])
            score.distance.append(diff_alt)
            # print(diff_alt)
    if not (test):
        show_policy(
            episode.thrust_values,
            episode.theta_values,
            episode.x_values,
            episode.z_values,
            score.distance,
            combination,
            title="pvt_train_" + str(batch),
        )
    if test:
        show_policy(
            episode.thrust_values,
            episode.theta_values,
            episode.x_values,
            episode.z_values,
            score.distance,
            combination,
            title="pvt_" + str(batch),
        )
        if not os.path.exists(os.path.join("env", "Pos_and_angles", str(batch))):
            os.mkdir(os.path.join("env", "Pos_and_angles", str(batch)))
        write_pos_and_angles_to_txt(environment, "Pos_and_angles/" + str(batch))
    plot_multiple(
        Series=[score.reward, score.reward_mean],
        labels=["Reward", "Mean reward"],
        xlabel="time (s)",
        ylabel="Reward",
        title="Global Reward vs time",
        save_fig=True,
        path="env",
        folder=str(combination),
    )
    return environment.FlightModel.Pos[0]


def batch_information(
    i, result_vec, combination, total_combination, temp_time, number_batches
):
    if result_vec:

        print(
            "Combination {}/{}, Batch {}/{}, Best result: {},Time per batch {}s, Combination ETA: {}mn{}s, Total ETA: {}mn{}s".format(
                combination,
                total_combination,
                i,
                number_batches,
                int(result_vec[-1]),
                round(temp_time / i, 1),
                round(((temp_time * number_batches / i) - temp_time) // 60),
                round(((temp_time * number_batches / i) - temp_time) % 60),
                round(((temp_time * number_batches / i) * total_combination) // 60),
                round(((temp_time * number_batches / i) * total_combination) % 60),
            )
        )


def trainer(
    environment,
    agent,
    max_step_per_episode,
    n_episodes,
    n_episodes_test=1,
    combination=1,
    total_combination=1,
):

    result_vec = []
    start_time = time.time()
    number_batches = round(n_episodes / 100) + 1
    for i in range(1, number_batches):
        temp_time = time.time() - start_time
        batch_information(
            i, result_vec, combination, total_combination, temp_time, number_batches
        )
        # Train agent
        run(
            environment,
            agent,
            100,
            max_step_per_episode,
            combination=combination,
            total_combination=total_combination,
            batch=i,
        )
        # Test Agent
        result_vec.append(
            run(
                environment,
                agent,
                n_episodes_test,
                max_step_per_episode,
                combination=combination,
                total_combination=total_combination,
                batch=i,
                test=True,
            )
        )
    environment.FlightModel.plot_graphs(save_figs=True, path="env")
    plot_multiple(
        Series=[result_vec],
        labels=["TO-Distance"],
        xlabel="episodes",
        ylabel="Distance (m)",
        title="TO-Distance vs episodes",
        save_fig=True,
        path="env",
        folder=str(combination),
        time=False,
    )
    agent.close()
    environment.close()
    save_distances(
        result_vec, combination, environment
    )  # saves distances results for each combination in a txt file.
    return environment.FlightModel.Pos[0]


def save_distances(result_vec, combination, environment):
    """
    Saves distances results in a txt in the current combination folder
    """
    if not os.path.exists(os.path.join("env", "Distances", str(combination))):
        os.mkdir(os.path.join("env", "Distances", str(combination)))
    write_to_txt_general(result_vec, "Distances/" + str(combination) + "/distances.txt")
    write_pos_and_angles_to_txt(environment, "Distances/" + str(combination))

