from unityagents import UnityEnvironment
import numpy as np
from ddpg_agent import Agent
import random
import torch
from collections import deque
import matplotlib.pyplot as plt

env = UnityEnvironment(file_name="Tennis.exe")
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

def ddpg(n_episodes=50000, max_t=3000):
    scores = []
    scores_window = deque(maxlen=100)
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        agent.reset()
        episode_score = np.zeros(num_agents)
        t=0
        while True:
            t+=1
            action=agent.act(states)
            env_info=env.step(action)[brain_name]
            next_state = env_info.vector_observations
            reward=env_info.rewards
            done=env_info.local_done
            agent.step(states[0], action[0], reward[0], next_state[0], done[0],t)
            agent.step(states[1], action[1], reward[1], next_state[1], done[1],t)
            states=next_state
            episode_score += reward
            if np.any(done):
                break
        max_score=np.max(episode_score)
        scores_window.append(max_score)
        scores.append(max_score)
        print('\rEpisode {}\tAvg. score: {:.2f}\t'.format(i_episode, np.mean(scores_window), end=""))
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAvg. score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=0.5:
            print('\nSolved in {:d} episodes.\tAvg score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            break
    return scores

agent = Agent(state_size=state_size, action_size=action_size, random_seed=1)
historized_scores = ddpg()

# plot results:
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(historized_scores)+1), historized_scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
env.close()