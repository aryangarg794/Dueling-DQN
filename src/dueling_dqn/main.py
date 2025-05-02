import argparse
import ale_py
import torch 
import numpy as np
import random
import gymnasium as gym
import matplotlib.pyplot as plt

from dueling_dqn.model.arch import DuelingNetwork
from dueling_dqn.utils.train import train, plot_results, make_env

gym.register_envs(ale_py)

parser = argparse.ArgumentParser(description="Dueling DQN Experiment Runner")

parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="List of random seeds for running multiple experiments")
parser.add_argument("--env", type=str, default='CartPole-v1', help="Env to test on")
parser.add_argument("--save", action="store_true", help="Save model or not")
parser.add_argument("--steps", type=int, default=int(1e6), help="How long to run the experiment")
parser.add_argument("--val", type=int, default=5000, help="When to evaluate")
parser.add_argument("--pre", type=int, default=int(25e3), help="How much to preload")
parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning Rate")
parser.add_argument("--atari", action="store_true", help="Atari game or not")
parser.add_argument("--buffer", type=str, default='prop', help="Buffer type")
parser.add_argument("--batch_size", type=int, default=1024, help="Batch Size")

args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    results = []
    best_score = float('-inf')
    
    print(f'================Running env: {args.env} for {args.steps} steps================')
    for seed in args.seeds:
        print(f'\n==============Running Seed {seed}================\n')
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    
        env = make_env(args.env, args.atari)
        env.action_space.seed(seed)
        
        agent = DuelingNetwork(
            env, lr=args.lr, buffer_type=args.buffer, atari=args.atari
        )
        
        metrics = train(
            env, agent, args.steps, args.val, args.batch_size,
            args.pre, seed=seed
        )
        
        results.append(metrics.averages)
        
        # save the best model
        if metrics.get_average > best_score and args.save: 
            agent.save(args.env)
            best_score = metrics.averages
            
    results = np.array(results)
    plot_results(results, args.steps, args.val, args.env, True)