import gymnasium as gym
import numpy as np 
import matplotlib.pyplot as plt
import torch 

from tqdm import tqdm
from typing import List, Tuple
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
from gymnasium.wrappers import FrameStackObservation

from dueling_dqn.model.arch import DuelingNetwork
from dueling_dqn.utils.metrics import RollingAverage

def train(
    env: gym.Env, 
    agent: DuelingNetwork,  
    timesteps: int = 1000000, 
    val_freq: int = 5000, 
    batch_size: int = 1024, 
    preload: int = 1000, 
    window: int = 5, 
    num_val_runs: int = 10,
    seed: int = 0
) -> RollingAverage: 
    
    metrics = RollingAverage(window)

    obs, _ = env.reset(seed=seed)
    done = False
    for _ in tqdm(range(preload)):
        action = env.action_space.sample()
        obs_prime, reward, terminated, truncated, _ = env.step(action)
        
        done = terminated or truncated
        transition = (obs, action, reward, obs_prime, done)
        agent.buffer.add(transition)
        
        obs = obs_prime
        if done: 
            obs, _ = env.reset(seed=seed)
            done = False
    
    obs, _ = env.reset(seed=seed)
    done = False
    
    print('\n')
    for step in range(1, timesteps+1):
        action = agent.epsilon_greedy(obs)
        obs_prime, reward, terminated, truncated, _ = env.step(action)
        
        done = terminated or truncated
        transition = (obs, action, reward, obs_prime, done)
        agent.buffer.add(transition)
        
        obs = obs_prime
        if done:
            obs, _ = env.reset(seed=seed)
            done = False

        # sample and update
        batch = agent.buffer.sample(batch_size)
        agent.update(batch)
        agent.soft_update()
        
        if step % val_freq == 0 or step == 1:
            val_reward = agent.evaluate(num_val_runs)
            metrics.update(val_reward)
            print(f'Timestep: {step} | Average Val Reward: {metrics.get_average:.4f}', end='\r')
        
    env.close()
    agent.val_env.close()
        
def plot_results(scores, timesteps, val_freq, game_name, save: bool = False):
    vars_low = []
    vars_high = []
    q=10

    for i in range(scores.shape[1]):
        vars_low.append(np.percentile(scores[:, i], q=q))
        vars_high.append(np.percentile(scores[:, i], q=100-q))

    mean_scores = np.mean(scores, axis=0) 
    plt.style.use('ggplot')   
    
    color = 'r'
    xs = np.arange(0, timesteps+val_freq, val_freq)
    plt.plot(xs, mean_scores, label='Average Val Score', color=color)
    plt.plot(xs, vars_low, alpha=0.1, color=color)
    plt.plot(xs, vars_high, alpha=0.1, color=color)
    plt.fill_between(xs, vars_low, vars_high, alpha=0.2, color=color)
    plt.legend()
    plt.grid(True)
    plt.title(f'{game_name} Dueling DQN + PER Results')
    plt.ylabel('Cumm. Reward')
    plt.xlabel('Timestep')
    if save:
        plt.savefig(f'lcs/dueling_dqn_{game_name}')
        
def make_env(game_name: str, atari: bool) -> gym.Env:
    env = gym.make(game_name)
    if atari: 
        env = AtariPreprocessing(env)
        env = FrameStackObservation(env, stack_size=4)
    return env 