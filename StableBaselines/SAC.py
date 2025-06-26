import gymnasium as gym

from stable_baselines3 import SAC

env = gym.make("InvertedDoublePendulum-v5", render_mode="human")

model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=300 * 1e6, progress_bar=True)

# vec_env = model.get_env()
# obs = vec_env.reset()
# while True:
#     action, _state = model.predict(obs, deterministic=True)
#     obs, reward, done, info = vec_env.step(action)
#     vec_env.render("human")
#     # VecEnv resets automatically
#     if done:
#       obs = vec_env.reset()