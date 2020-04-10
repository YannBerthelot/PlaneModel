import os
import time
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
    sum_rewards = 0.0
    environment.FlightModel.timestep_max = max_step_per_episode
    for i in range(n_episodes):
        temp_time = time.time() - start_time
        time_per_episode = temp_time / (i + 1)
        print(
            "episode : ",
            i,
            "time per episode",
            round(time_per_episode, 2),
            "estimated time to finish",
            int(time_per_episode * n_episodes),
        )
        # Initialize episode
        states = environment.reset()
        internals = agent.initial_internals()
        terminal = False

        while not terminal:
            # Episode timestep
            actions = agent.act(states=states)

            states, terminal, reward = environment.execute(actions=actions)

            agent.observe(terminal=terminal, reward=reward)
            sum_rewards += reward
            if terminal:
                print(FlightModel.action_vec[actions], FlightModel.timestep)
                print(states, terminal, reward)

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
    reward_estimation=dict(horizon=1000),
)

# Define train parameters
max_step_per_episode = 1000
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
