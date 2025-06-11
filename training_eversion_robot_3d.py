import gymnasium as gym
from stable_baselines3 import PPO, DQN
from eversion_robot_3d import EversionRobot3D
from stable_baselines3.common.callbacks import EvalCallback

# Create environment
env = EversionRobot3D(obs_use=True)
eval_env = EversionRobot3D(obs_use=True)

eval_callback = EvalCallback(
    eval_env,
    eval_freq=100000,  # evaluate every 100k steps
    log_path="./ppo_eval_logs/",
    best_model_save_path="./ppo_best_model/",
    deterministic=True,
)
# Wrap the environment in a DummyVecEnv for vectorized training
model = PPO(
    "MlpPolicy",
    env,
    verbose=2,
    tensorboard_log="./PPO_eversion_obs_tensorboard/",
    seed=42,
)

model.learn(
    total_timesteps=50000000, tb_log_name="eversion_run", callback=eval_callback
)

# Save the model
model.save("PPO_eversion")
print("Done")
