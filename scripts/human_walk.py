import gymnasium as gym
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env


# Function to create a single environment instance
def make_env(env_id, rank, seed=0):
    def _init():
        env = gym.make(env_id, render_mode="rgb_array")  # Create a Gym environment
        env.reset(seed=seed + rank)  # Reset the environment with a seed
        return env
    return _init

# Function to train and evaluate RL
def train_and_evaluate(env_id, num_envs, total_timesteps, n_eval_episodes, algorithm):
    # Create parallel environments using make_vec_env
    envs = SubprocVecEnv([make_env(env_id, i) for i in range(num_envs)])
    
    # Wrap environments with VecMonitor for monitoring
    envs = VecMonitor(envs)

    # Instantiate the agent
    model = algorithm(
        "MlpPolicy",  # Use Multi-Layer Perceptron policy
        env=envs,  # Use parallel environments
        verbose=1,  # Output training progress
        tensorboard_log=f"./{algorithm.__name__}_train_rewards"  # Tensorboard log directory
    )

    # Train the agent, display a progress bar and define tensorboard name
    model.learn(total_timesteps=total_timesteps, progress_bar=True, tb_log_name=f"{algorithm.__name__}_"+"first_run")

    # Save the trained agent
    model.save(f"../trained_models/{algorithm.__name__}_" + env_id)
    del model  # Delete the trained model to demonstrate loading

    # Load the trained agent
    model = algorithm.load(f"../trained_models/{algorithm.__name__}_" + env_id, env=envs)

    # Evaluate the agent
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=n_eval_episodes)
    print(f"{algorithm.__name__}: Mean reward: {mean_reward} +/- {std_reward}")

if __name__ == "__main__":
    from stable_baselines3 import PPO, SAC, A2C  # Import algorithms PPO, SAC, A2C

    # Define the number of parallel environments
    num_envs = 40  
    env_id = "Humanoid-v3"
    total_timesteps = int(1e8)  # Number of training steps
    n_eval_episodes = 10  # Number of evaluation episodes

    # Uncomment one of the following lines to select the algorithm to use
    train_and_evaluate(env_id, num_envs, total_timesteps, n_eval_episodes, PPO)
    #train_and_evaluate(env_id, num_envs, total_timesteps, n_eval_episodes, SAC)
    #train_and_evaluate(env_id, num_envs, total_timesteps, n_eval_episodes, A2C)
