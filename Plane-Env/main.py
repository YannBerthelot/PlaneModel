import os
import time
import numpy as np
from env.FlightModel import FlightModel
from env.FlightEnv import PlaneEnvironment
from utils import write_to_txt
from env.AnimatePlane import animate_plane
from tensorforce import Agent, Environment


def train(n_episodes, max_step_per_episode):
    """
    Train agent for n_episodes
    """
    start_time = time.time()

    environment.FlightModel.max_step_per_episode = max_step_per_episode
    for i in range(n_episodes):
        sum_rewards = 0.0
        episode_length = 0
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
        # Initialize episode
        states = environment.reset()
        internals = agent.initial_internals()
        terminal = False
        thrust_vec = []
        theta_vec = []
        while not terminal:
            episode_length += 1
            # Episode timestep
            actions = agent.act(states=states)
            thrust_vec.append(round(FlightModel.action_vec[actions][0], 2))
            theta_vec.append(round(FlightModel.action_vec[actions][1], 2))

            states, terminal, reward = environment.execute(actions=actions)

            agent.observe(terminal=terminal, reward=reward)
            sum_rewards += reward
            if terminal:
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

    end_time = time.time()
    total_time = end_time - start_time
    print("total_time", total_time)


# Instantiate our Flight Model
FlightModel = FlightModel()
# Instantiane our environment
environment = PlaneEnvironment()
# Instantiate a Tensorforce agent
agent = Agent.create(
    agent="tensorforce",
    environment=environment,  # alternatively: states, actions, (max_episode_timesteps)
    memory=100000,
    update=dict(unit="timesteps", batch_size=64),
    optimizer=dict(type="adam", learning_rate=1e-3),
    policy=dict(network="auto"),
    objective="policy_gradient",
    reward_estimation=dict(horizon=1000, estimate_actions=True),
    exploration=0.05,
)

# Define train parameters
max_step_per_episode = 600
n_episodes = 50
# Train agent
train(n_episodes, max_step_per_episode)

# Save last run positions
write_to_txt(environment)
# Animate last run positions
animate_plane()


# End
agent.close()
environment.close()
