import argparse
import ale_py
import os
import numpy as np
import random
import gymnasium as gym
import torch 
import dill


from dueling_dqn.model.arch import DuelingNetwork
from dueling_dqn.utils.train import train, plot_results, make_env

gym.register_envs(ale_py)

parser = argparse.ArgumentParser(description="Dueling DQN Experiment Runner")

parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4], help="List of random seeds for running multiple experiments")
parser.add_argument("--env", type=str, default='CartPole-v1', help="Env to test on")
parser.add_argument("--save", action="store_true", help="Save model or not")
parser.add_argument("--steps", type=int, default=int(1e6), help="How long to run the experiment")
parser.add_argument("--val", type=int, default=250, help="When to evaluate")
parser.add_argument("--pre", type=int, default=int(25e3), help="How much to preload")
parser.add_argument("--decay_steps", type=int, default=int(1e5), help="When to stop decay")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate")
parser.add_argument("--atari", action="store_true", help="Atari game or not")
parser.add_argument("--buffer", type=str, default='rank', help="Buffer type")
parser.add_argument("--buffer_size", type=int, default=int(1e5), help="Buffer Size")
parser.add_argument("--batch_size", type=int, default=512, help="Batch Size")

args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    results = []
    best_score = float('-inf')
    
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f'================Running env: {args.env} for {args.steps} steps on device: {device} for buffer {args.buffer}================')
    for seed in args.seeds:
        print(f'\n==============Running Seed {seed}================\n')
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    
        env = make_env(args.env, args.atari)
        env.action_space.seed(seed)
        
        agent = DuelingNetwork(
            env, lr=args.lr, buffer_type=args.buffer, atari=args.atari, 
            decay_steps=args.decay_steps, stop_anneal=args.steps-100000,
            buffer_size=args.buffer_size
        )
        
        try:
            metrics = train(
                env, agent, args.steps, args.val, args.batch_size,
                args.pre, seed=seed
            )
        except KeyboardInterrupt:
            env.close()
            agent.val_env.close()
        
        results.append(metrics.averages)
        
        # save the best model
        if metrics.get_average > best_score and args.save: 
            agent.save(args.env)
            best_score = metrics.get_average
            
    results = np.array(results)
    plot_results(results, args.steps, args.val, args.env, args.buffer, True)
    with open(f'lcs/results_{args.env}_{args.buffer}.pl', 'wb') as file: 
        dill.dump(results, file)