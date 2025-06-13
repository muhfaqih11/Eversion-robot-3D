from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from eversion_robot_3d import EversionRobot3D
from stable_baselines3.common.evaluation import evaluate_policy

# 1. Create and wrap the environment
env = DummyVecEnv([lambda: EversionRobot3D(obs_use=True)])

# 3. Test the model
obs = env.reset()
real_env = env.envs[0]  # Access the underlying environment
model = PPO.load("PPO_eversion_R80", env=env)  #

# Evaluate the policy
mean_reward, std_reward = evaluate_policy(
    model, env, n_eval_episodes=10, render=False, deterministic=False
)

print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

for _ in range(200):
    action, _ = model.predict(obs)  # Use deterministic actions for evaluation
    obs, reward, done, info = env.step(action)
    real_env.render()  # Render the environment
    # Access custom attributes from the real environment
    print(f"State: {real_env.state}, Action: {real_env.act}, Reward: {reward}")

    if done:
        obs = env.reset()
