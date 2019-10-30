import numpy as np
from agents import Agent
from state import Environment
from simulation import Simulation


# For post-analysis
pi_historical = []
w_historical = []
r_historical = []
t_historical = []

# For intra-algo
mu_bar_historical = []
mu_historical = []

# Init the Agent's environment
env = Environment()

# Init the expert agent
# Feed it the expert trajectories
a = Agent(type='expert',
          action_list=['l', 'r'],
          environment=env,
          trajectories=[['r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r']])

# Build said expert trajectories
a.build_trajectories()

# Build the Agent's initial state distribution
a.build_D()

# Init a standalone environment for the state itself
simul_env = Environment()

# Init the simulation
sim = Simulation(agents=a, environment=simul_env, alpha=.02)

# This method will initalize a matrix of state-action pairs and their values (currently set to init all
# at 0). This  will build a matrix that represents all of the states that the expert agent has visited
sim.reset_q(trajectories=sim.agents['expert'].state_trajectories)

# W
w = np.array([-0.5, 0.5])

cumsum_r_ts = sim.q_learning(break_condition=.00001,
                             num_steps=10,
                             epsilon=.1,
                             w=w,
                             gamma=0.99,
                             IRL='test')

print(sim.Q)
print(sim.state_q_mapping)

greedy = sim.greedy_policy(i=10)
print(greedy)
