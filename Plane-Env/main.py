from env.FlightModel import FlightModel
from env.FlightEnv import PlaneEnvironment
from utils import gridsearch_tensorforce
from env.AnimatePlane import animate_plane


def main(animate=False):
    # Instantiane our environment
    environment = PlaneEnvironment()
    # Instantiate a Tensorforce agent
    param_grid_list = {}
    param_grid_list["PPO"] = {
        "batch_size": [5],
        "update_frequency": [50],
        "learning_rate": [1e-3],
        "subsampling_fraction": [0.3],
        "optimization_steps": [10, 30, 50],
        "likelihood_ratio_clipping": [0.1],
        "discount": [0.99],
        "estimate_terminal": [False],
        "multi_step": [30],
        "learning_rate_critic": [1e-3],
        "exploration": [0.01],
        "variable_noise": [0.0],
        "l2_regularization": [0.001],
        "entropy_regularization": [0.01],
        "network": ["auto"],
        "size": [64],
        "depth": [4],
    }

    gridsearch_tensorforce(
        environment, param_grid_list["PPO"], max_step_per_episode=1000, n_episodes=10000
    )

    # Animate last run positions
    if animate:
        animate_plane()


if __name__ == "__main__":
    main(animate=False)

