# Proximal Policy Optimization (PPO) for Chemical Reactor Control

This project implements a **Proximal Policy Optimization (PPO)** reinforcement learning agent for controlling a chemical batch reactor to track time-varying temperature setpoints. The agent learns to optimize the jacket temperature (cooling/heating) to maintain the reactor temperature close to a reference trajectory while minimizing control effort.

## Project Overview

The implementation uses PPO, a state-of-the-art policy gradient method, to solve the reactor temperature control problem. The reactor system involves:
- **State Variables**: Time step, reactor temperature (T), and reactant concentration (Ca)
- **Action**: Jacket temperature (Tj) selection from a discretized range
- **Objective**: Track a time-varying reference temperature trajectory while minimizing temperature deviation and control changes

## Features

- **PPO Algorithm**: Actor-critic architecture with clipped surrogate objective
- **Neural Network Models**: 
  - Actor network (policy) with softmax output for action selection
  - Critic network (value function) for advantage estimation
- **Generalized Advantage Estimation (GAE)**: For variance reduction
- **Replay Memory**: Experience storage and batch sampling
- **Dynamic Visualization**: Real-time episode vs. reward plotting
- **Model Persistence**: Save and load trained models
- **Chemical Reactor Simulation**: Realistic batch reactor environment with:
  - Non-linear reaction kinetics
  - Temperature-dependent reaction rates (Arrhenius equation)
  - Time-varying reference trajectories

## Dependencies

The project requires the following Python libraries:
```
numpy
pandas
matplotlib
seaborn
tensorflow (2.x)
tensorflow-probability
openpyxl  # for Excel file handling
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/KalooriShiva/Proximal-Policy-Optimization.git
cd Proximal-Policy-Optimization
```

2. Install dependencies:
```bash
pip install numpy pandas matplotlib seaborn tensorflow tensorflow-probability openpyxl
```

## Usage

### Training the Agent

Run the main training script:
```bash
python main.py
```

The script will:
1. Initialize the reactor environment with 40 timesteps
2. Create a PPO agent with a neural network architecture [400, 300, 200]
3. Train for 70,000 episodes
4. Save the trained actor network as `A_network.h5`
5. Generate visualization plots saved as a PDF file with timestamped name
6. Print performance metrics (MAE, RMSE, training time)

### Key Parameters

You can modify parameters in `main.py`:
- `timesteps`: Number of time steps in each episode (default: 40)
- `num_j_temp`: Number of discrete jacket temperature actions (default: 40)
- `learning_rate`: Learning rate for the optimizer (default: 1e-6)
- `nn_arch`: Neural network architecture layers (default: [400, 300, 200])
- Number of training episodes (default: 70,000)

### Reactor Environment Configuration

In `reactor_environment.py`, you can adjust:
- `batch_time`: Total time for batch process (default: 80 minutes)
- `min_temp`, `max_temp`: Reactor temperature bounds (default: 293-308 K)
- `min_j_temp`, `max_j_temp`: Jacket temperature bounds (default: 273-318 K)
- `k0`: Pre-exponential factor for reaction kinetics

## Project Structure

```
.
├── main.py                    # Main training and evaluation script
├── agent.py                   # PPO agent implementation with actor-critic
├── reactor_environment.py     # Chemical reactor environment simulation
├── batch_nmpc.m              # MATLAB implementation of batch NMPC (reference)
├── Sample_data.xlsx          # Sample trajectory data
├── Using Replay memory/      # Alternative implementations
│   ├── agent.py
│   ├── main.py
│   └── reactor_environment
└── README.md                 # This file
```

## Output

The program generates:
1. **Trained Model**: `A_network.h5` - The trained actor network
2. **Plots PDF**: Contains four plots:
   - Reactor temperature vs. reference trajectory
   - Jacket temperature (control action) over time
   - Reactant concentration over time
   - Episode rewards with rolling average
3. **Console Output**: MAE, RMSE, and training time statistics

## Algorithm Details

### PPO (Proximal Policy Optimization)
- **Clipping Parameter**: 0.2 (prevents large policy updates)
- **Discount Factor**: 1.0
- **GAE Lambda**: 0.95
- **Entropy Coefficient**: 0.001 (encourages exploration)
- **Batch Size**: 20 samples from replay memory
- **Optimizer**: RMSprop with learning rate decay

### Reward Function
The reward at each timestep is calculated as:
```
r = -(Q_T * (T - T_ref)² + Q_Ca * Ca² + R * ΔTj²)
```
where:
- Q_T, Q_Ca: State penalty weights
- R: Control change penalty weight
- ΔTj: Change in jacket temperature

## Performance Metrics

The system evaluates performance using:
- **MAE** (Mean Absolute Error): Average absolute deviation from reference
- **RMSE** (Root Mean Square Error): Square root of mean squared deviations
- **Episode Reward**: Cumulative reward per episode

## References

This implementation is related to reinforcement learning approaches for chemical process control:
- Proximal Policy Optimization: [Schulman et al., 2017](https://arxiv.org/abs/1707.06347)
- Time-varying setpoint tracking for batch processes
- Model Predictive Control (MPC) comparison baseline

## License

Please refer to the repository license file for usage terms.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## Author

KalooriShiva

## Acknowledgments

This project demonstrates the application of deep reinforcement learning to chemical process control, specifically batch reactor temperature tracking problems.