import gymnasium as gym
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
#import multiprocessing as mp

# 确保在所有操作系统上使用 'spawn' 启动方法
# mp.set_start_method("spawn")

# 创建一个环境的函数
def make_env(env_id, rank, seed=0):
    def _init():
        env = gym.make(env_id, render_mode="rgb_array")
        env.reset(seed=seed + rank)
        return env
    return _init

def train_and_evaluate(env_id, num_envs, total_timesteps, n_eval_episodes, algorithm):
    # 使用 make_vec_env 创建并行环境
    envs = SubprocVecEnv([make_env(env_id, i) for i in range(num_envs)])
    
    # 使用 VecMonitor 包装环境
    envs = VecMonitor(envs)

    # 实例化智能体
    model = algorithm(
        "MlpPolicy",
        env=envs,
        verbose=1,
        tensorboard_log=f"./{algorithm.__name__}_train_rewards"
    )

    # 训练智能体并显示进度条
    model.learn(total_timesteps=total_timesteps, progress_bar=True, tb_log_name=f"{algorithm.__name__}_"+"first_run")

    # 保存智能体
    model.save(f"{algorithm.__name__}_" + env_id)
    del model  # 删除训练好的模型以演示加载

    # 加载训练好的智能体
    model = algorithm.load(f"{algorithm.__name__}_" + env_id, env=envs)

    # 评估智能体
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=n_eval_episodes)
    print(f"{algorithm.__name__}: Mean reward: {mean_reward} +/- {std_reward}")

if __name__ == "__main__":
    from stable_baselines3 import PPO, SAC, A2C   # 这里可以替换为其他算法，如 PPO, SAC, A2C 等

    # 定义并行环境的数量
    num_envs = 40  
    env_id = "Humanoid-v3"
    total_timesteps = int(1e7)  # 训练步数
    n_eval_episodes = 10  # 评估次数

    train_and_evaluate(env_id, num_envs, total_timesteps, n_eval_episodes, PPO)
    #train_and_evaluate(env_id, num_envs, total_timesteps, n_eval_episodes, SAC)
    #train_and_evaluate(env_id, num_envs, total_timesteps, n_eval_episodes, A2C)
