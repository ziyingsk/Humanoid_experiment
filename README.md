
# Humanoid Reinforcement Learning Experiment

This is a small experiment using reinforcement learning to make a humanoid model walk, aiming to gain some experience and intuition.

## Introduction

This reinforcement learning experiment primarily uses [Humanoid-V3](https://www.gymlibrary.dev/environments/mujoco/humanoid/) from OpenAI's Gym and [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) (sb3). According to the sb3 documentation, the default parameters of its reinforcement learning algorithms are somewhat empirical and can be used directly in some cases. Therefore, I was curious if this applies to robot control as well. Here, I compared three algorithms: PPO, SAC, and A2C.

While using these algorithms, I only specified the necessary parameters, and the action policy was consistently `MlpPolicy`. During training, I used 2 V100 GPUs and conducted 10M steps for each algorithm. The training took about 4 hours.

## Results

Here is the graph showing the relationship between reward and training steps: ![Reward vs Training Steps](./train_rewards/reward.png)

Results of the three algorithms:
- **PPO**: [Video](./videos/PPO.mp4)
- **A2C**: [Video](./videos/A2C.mp4)
- **SAC**: [Video](./videos/SAC.mp4)

## Discussion

First of all, it must be emphasized that this is just a simple experiment, so the following discussion is just some intuition:

Let's review the key differences between these three algorithms:
- **PPO**: Utilizes clipping to restrict overly rapid changes in action policies, ensuring stability in learning.
- **A2C**: Uses statistical methods to reduce the variance of Q-values without changing the expectation by subtracting a baseline with zero expectation.
- **SAC**: Adds an entropy term to the action policy to encourage more exploration.

From this, we can see that A2C and PPO emphasize learning stability, while SAC emphasizes exploration. In other words, the humanoid posture control task involves a more complex dimensional space compared to many other tasks suitable for default parameters. Therefore, humanoid posture control relatively requires more emphasis on exploration in learning.

## Interesting Future Directions

From the videos, we can see that although SAC achieved walking, its walking style is very peculiar. Making it walk more like a human would be very interesting. Currently, I see two promising directions:
1. Modify the reward function so that human-like walking postures receive higher rewards compared to other walking styles.
2. Use imitation learning to mimic human postures. In this way, it becomes similar to ChatGPT, where we are essentially using RLHF, which is fascinating!

## Reproducing the Experiment

If you want to reproduce my experiment using my code, it's in the `script` folder. Due to compatibility issues between MuJoCo and OpenAI Gym, I encountered some difficulties getting the program to run properly. Therefore, even with the provided `requirements.txt`, you may need to do some debugging to get the program to run successfully. Feel free to raise any issues you encounter, and if I have dealt with the same problems, I might be able to provide solutions.

### File Descriptions

1. **requirements.txt**

For creating the virtual environment. I used Python version 3.10. Here are some configuration suggestions:
- Avoid using Windows OS.
- You may need to manually load the swig package, as this dependency is not clearly stated in the library.
- If you encounter errors with MuJoCo, this site is very helpful: [MuJoCo-py](https://github.com/openai/mujoco-py?tab=readme-ov-file)
- If you encounter graphical display errors, the following commands helped me:
    ```bash
    sudo apt-get update
    sudo apt-get install -y python3-opengl
    apt install ffmpeg
    apt install xvfb
    pip3 install pyvirtualdisplay
    ```

2. **human_walk.py**

Used for training the model. The current program uses parallel computing. If your device cannot use GPU parallelism, you will get errors and may need to remove the parallelism in the code. You may receive some warning messages during training, but you can ignore them. The code integrates tensorboard (if not automatically created, you may need to create a log folder), and during training, you can use [tensorboard](https://stable-baselines3.readthedocs.io/en/v2.1.0_a/guide/tensorboard.html) to monitor and collect training data.

3. **human_walk_video.py**

Used to generate videos with the trained model. You need to create a `videos` folder before running the code. The code defaults to loading the model trained with the PPO algorithm.
