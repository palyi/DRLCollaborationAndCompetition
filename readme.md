
[![IMAGE ALT TEXT HERE](http://img.youtube.com/vi/GNtxjSdOQ9w/0.jpg)](http://www.youtube.com/watch?v=GNtxjSdOQ9w)

<h2>Project Details</h2>
<p>
In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.
<br>
The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.
<br>
The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,
<br>
After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
This yields a single score for each episode.
<br>
The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.
</p>
<h2>Getting Started</h2>
<p>
For this project, you will not need to install Unity - this is because an environment is already built, and you can download it from the link below (64-bit Windows):
https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip
</p>
<h2>Instructions</h2>
<p>I built this solution on the 'frame' of the previous project entirely. The model is re-used with some very minor neuron number modifications, and so is the agent with slightly modified hyperparameters. The training process takes can be launched by running tennis.py. To watch the agent in action, please run test_trained.py.</p>
<h2>Installing Dependencies and downloading needed files.</h2>
<p>
1. Install Anaconda and create an environment<br>
2. Install pyTorch by pip install torch===1.4.0 torchvision===0.5.0 -f https://download.pytorch.org/whl/torch_stable.html
  This command lets you install pyTorch via Pip, in a Windows environment, utilizing CUDA 10.1 (commands for other environments can be generated at pyTorch.org<br>
3. Install Unity Machine Learning Agents by pip install mlagents<br>
4. The environment is provided by a Windows (64-bit) executable, which can be downloaded from: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip<br>
</p>
