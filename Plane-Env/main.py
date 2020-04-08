import gym
import json
import datetime as dt
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from env.FlightModel import FlightModel
from env.FlightEnv import PlaneEnvironment

FlightModel = FlightModel()
# env = DummyVecEnv(PlaneEnvironment())
env = PlaneEnvironment()
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from rl.agents.cem import CEMAgent
from rl.memory import EpisodeParameterMemory

nb_actions = PlaneEnvironment().NUM_ACTIONS
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(nb_actions))
model.add(Dense(nb_actions))
model.add(Dense(nb_actions))
model.add(Dense(nb_actions))
model.add(Activation("softmax"))
obs = env.reset()
print(model.summary())

memory = EpisodeParameterMemory(limit=1000, window_length=1)

cem = CEMAgent(
    model=model,
    nb_actions=nb_actions,
    memory=memory,
    nb_steps_warmup=2000,
    train_interval=50,
    elite_frac=0.05,
)
cem.compile()

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
cem.fit(env, nb_steps=100000, visualize=False, verbose=1)

# After training is done, we save the best weights.
cem.save_weights("cem_{}_params.h5f".format("PlaneEnv"), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
cem.test(env, nb_episodes=5, visualize=True)

# for i in range(2000):
#     print(obs,env.observation_space.shape)
#     action, _states = model.predict(obs)
#     obs, rewards, done, info = env.step(action)
#

