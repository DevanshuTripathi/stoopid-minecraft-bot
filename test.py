from stable_baselines3 import PPO
from minecraft_env import MinecraftDragonEnv
import time

# Initialize the environment
env = MinecraftDragonEnv()

# Load the trained model
# Make sure the name matches what you saved it as in train.py
model = PPO.load("dragon_ppo_1", device="cpu")

obs, info = env.reset()
done = False

print("Starting evaluation. Open http://localhost:3007 in your browser to watch!")

while not done:
    # Use deterministic=True for evaluation to pick the absolute best action
    action, _states = model.predict(obs, deterministic=True)
    
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    # Optional: Slow down the python loop slightly if it desyncs from the 250ms JS steps
    time.sleep(0.05)

print("Episode finished!")
env.close()