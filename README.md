# Computer Vision Course Project #
In this project, I implemented the Proximal Policy Optimization algorithm ([OpenAI website](https://openai.com/blog/openai-baselines-ppo/)) and analyze its performance on the MsPacman-v0 from openAI-GYM Atari environment using image inputs: [environment](https://gym.openai.com/envs/#atari) 
# Algorithm #
Proximal Policy Gradient (PPO) is an improved version of vanilla policy gradient in reinforcement learning. Compared to TRPO, which applied hard constraint to solve convergence problem, PPO applies soft constraint and strikes a balance between ease of implementation, sample complexity, and ease of tuning. It tries to compute an update at each step that minimizes the cost function while ensuring the deviation from the previous policy is relatively small. In the algorithm, I trained an actor and a critic using neural networks. The actor, where PPO is applied, is trained to choose the best action based on some policy given image input. The critic is trained to estimate the goodness of a specific game status. 
![algorithm](https://github.com/sai-shi/Reinforcement-Learning/blob/master/actor_critic.png)
# Improvement #
- Feature extraction using 5-layer CNN 
- Motion tracking: Using subtraction between sequential images as input 
- Epsilon-greedy: Choose action epsilon greedily and improve exploration 
# Result #
Trained robot (5000 episodes, 15 hours): 
![result](https://github.com/sai-shi/Reinforcement-Learning/blob/master/diff_cnn.gif)
