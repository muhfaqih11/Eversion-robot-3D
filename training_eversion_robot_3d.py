import gymnasium as gym
from stable_baselines3 import PPO, DQN
from eversion_robot_3d import EversionRobot3D

# Create environment
env = EversionRobot3D(obs_use=False)

# Set up DQN with TensorBoard logging
model = PPO("MlpPolicy", env, verbose=2, tensorboard_log="./PPO_eversion_tensorboard/")

# Train the model and log to TensorBoard under the "eversion_run" subdirectory
model.learn(total_timesteps=50000000, tb_log_name="eversion_run")

# Save the model
model.save("PPO_eversion_no_obs")
print("Done")
