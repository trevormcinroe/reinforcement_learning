from agents import Agent
from simulation import Simulation
from state import Environment

# Init the Agent's environment
env = Environment()

# Init the expert agent
# Feed it the expert trajectories
a = Agent(type='expert',
          action_list=['a', 'b', 'c', 'd', 'e', 'f', 'g'],
          environment=env,
          trajectories=[['a', 'b', 'c', 'e', 'g', 'b', 'c', 'e', 'g',],
                        ['a', 'b', 'c', 'a', 'g', 'g', 'a', 'g', 'g'],
                        ['c', 'd', 'f', 'b', 'c', 'a', 'd', 'f', 'b', 'c']])

# Build said expert trajectories
a.build_trajectories()

# Build the Agent's initial state distribution
a.build_D()

# Init a standalone environment for the state itself
simul_env = Environment()

# Init the simulation
sim = Simulation(agents=a, environment=simul_env)

# Need to initialize Q(s,a)



# TODO: FIX THIS METHOD
# TODO: it should return a list of lists. The inner list is a list of  dictionaries.





# This method will initalize a matrix of state-action pairs and their values (currently set to init all
# at 0). This  will build a matrix that represents all of the states that the expert agent has visited
sim.reset_q(trajectories=sim.agents['expert'].state_trajectories)


##################################
# === THE MAIN IRL ALGORITHM === #
##################################

# while t > BREAK_CONDITION (some arbitrarily small number)
# === (1a) Generate Random Policy === #
# This method helps to generate random trajectories. This is specifically used for the IRL algorithm
# NOTE: THE .build_trajectories() METHOD RESETS THE SIMULATION ENVIRONMENT WHEN DONE. NO NEED TO EXPLICITLY CALL IT
sim.gen_random_trajectory(i=15, n=1)
# IS THIS EVEN NEEDED?
rand_polic_state_trajs = sim.build_trajectories(trajectories=sim.random_policies)


# === (1b) Estimate μ for the random policy === #
# Now that weh have our random policy, estimate μ for it
# NOTE: THIS FUNCTION TAKES AN ACTION-TRAJECTORY, NOT A STATE TRAJECTORY
mu_random = sim.μ_estimate(trajectories=sim.random_policies, gamma=0.95)

# === (2) Projecton Method to estimate t, w === #
# === (3) Initiate RL algo to find optimal policy for R = w.T * ϕ === #
# === (4) Compute μ for the learned policy from the RL algo === #
