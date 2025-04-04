##  STEM Education Reinforcement Learning Agent (DQN/PG)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Stable-Baselines3](https://img.shields.io/badge/StableBaselines3-1.6.1-brightgreen)](https://stable-baselines3.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An AI agent that navigates a simulated STEM education environment to optimize student engagement using Deep Q-Networks (DQN) and Policy Gradient methods.

### ==> Environment Visualization 
![dqn_agent](https://github.com/user-attachments/assets/01615a66-5071-4beb-b43d-e262cbd68515)

### ==> Visual Elements:

Blue(🔵): Agent

Pink(🌸): Students

Green(🟢): Resources

Red(🔴): Barriers

Gray(⚫): Dropout triggers


##  Repository Structure 
```
caroline_gyireh_rl_summative/
├── environment                           # Environment
│ ├── stem_env.py                         # Environment logic
│ ├── rendering.py                        # PyOpenGL visualization
├── training/                             # RL algorithms
│ ├── dqn_training.py                     # DQN implementation
│ ├── pg_training.py                      # REINFORCE implementation
├── models/                               # Trained models
│ ├── dqn                                 # DQN models
│ └── pg                                  # Policy Gradient model
├── demo_video.py                         # Video demonstration
├── requirements.txt                      # Dependencies
└── README.md                             # This file, documentations
```

### Installations/SetUp
---------> Clone the repository
```bash
git clone https://github.com/cgyireh1/caroline_gyireh_rl_summative.git
```
---------> Go into the project directory
```bash
cd caroline_gyireh_rl_summative
```
---------> Install Dependencies
```bash
pip install -r requirements.txt
```

### Training The Agents
To Train both agents at once (DQN and Policy Gradient), use:
```bash
python main.py --train both
```
To Train a specific agent, use:
```bash
python main.py --train dqn or --train pg
```

### ==> Action Space

The Actions are discrete, and an agent can take any of these actions:
- Up
-  Down
- Left
- Right,
- Up-Left
- Up-Right
-  Down-Left
- Down-Right

### ==> State Space
- Empty
- Agent
- Student
- Barrier
- Resource
- Dropout

### ==>  Reward Structure

+50  for encouraging a student

+15  for collecting a resource

-5  for hitting a barrier

-20  for encountering a dropout

-0.1 per step penalty

**Termination: **
The Episode ends:
- When ** 3+ ** students are encouraged or
- After **200** steps are reached.


### Key Features
---> Custom STEM Environment:
- 10×10 grid with students, resources, and obstacles
- Multi-component observation space (grid + stats)
- Tunable reward structure

---> Algorithm Implementations:
- DQN with experience replay
- Policy Gradient with entropy regularization
- Hyperparameter optimization for both methods

*Success = Encouraging ≥3 students per episode*

