from training.dqn_training import train_dqn
from training.pg_training import train_pg
from analysis.metrics import full_analysis
import argparse

# 
def main():
    parser = argparse.ArgumentParser(description="STEM RL Training System")
    parser.add_argument('--train', choices=['dqn', 'pg', 'both'], default='both',
                      help="Which models to train")
    parser.add_argument('--analyze', action='store_true',
                      help="Run full analysis after training")
    args = parser.parse_args()
    
    # Training
    if args.train in ['dqn', 'both']:
        print("\n=== Training DQN ===")
        train_dqn()
    
    if args.train in ['pg', 'both']:
        print("\n=== Training Policy Gradient ===")
        train_pg()
    
    # Analysis
    if args.analyze:
        full_analysis()

if __name__ == "__main__":
    main()
