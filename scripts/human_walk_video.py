import gymnasium as gym
from stable_baselines3 import PPO, SAC, A2C
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import RecordVideo

# Define environment ID, specifying the Gym environment to use
env_id = "Humanoid-v3"

if __name__ == "__main__":
    # Create and wrap a single environment, Monitor is used to record environment information
    env = gym.make(env_id)
    env = Monitor(env)

    # Load the trained agent model, the model is trained using the PPO algorithm
    model = PPO.load("../trained_models/PPO_Humanoid-v3", env=env)

    # Set video recording parameters, specifying the folder to save the videos and video length
    video_folder = "videos"
    video_length = 2000

    # Create a new environment for displaying the trained agent, wrapped to enable video recording
    display_env = RecordVideo(
        gym.make(env_id, render_mode='rgb_array'), 
        video_folder=video_folder, 
        episode_trigger=lambda x: x == 0  # Trigger condition: only record the first episode
    )

    # Reset the environment and get the initial observation
    obs, info = display_env.reset()  # Get only the observation
    for i in range(video_length):
        # Use the model to predict actions, deterministic=True means using a deterministic policy
        action, _states = model.predict(obs, deterministic=True)
        # Execute the action and unpack all the information returned by the environment
        obs, rewards, dones, truncs, info = display_env.step(action)
        # Render the environment display
        display_env.render()
        # If the episode is done or truncated, reset the environment
        if dones or truncs:
            obs, info = display_env.reset()  # Get only the observation

    # Close the environment
    display_env.close()
