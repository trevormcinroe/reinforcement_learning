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
          trajectories=[['a', 'b', 'c', 'e', 'g', 'b', 'c', 'e', 'g', 'c'],
                        ['a', 'b', 'c', 'a', 'g', 'g', 'a', 'g', 'g', 'c'],
                        ['c', 'd', 'f', 'b', 'c', 'a', 'd', 'f', 'b', 'c']])

# Build said expert trajectories
a.build_trajectories()

# Build the Agent's initial state distribution
a.build_D()

# Init a standalone environment for the state itself
simul_env = Environment()

# Init the simulation
sim = Simulation(agents=a, environment=simul_env, alpha=1)

# Need to initialize Q(s,a)

# This method will initalize a matrix of state-action pairs and their values (currently set to init all
# at 0). This  will build a matrix that represents all of the states that the expert agent has visited
sim.reset_q(trajectories=sim.agents['expert'].state_trajectories)

# Generating the μ_e - the expert's feature expectations
mu_e = sim.μ_estimate(trajectories=sim.agents['expert'].trajectories, gamma=0.95)


##################################
# === THE MAIN IRL ALGORITHM === #
##################################
# As is explained in section 3.1 of "Apprenticeship Learning via IRL" by Abbeel, Ng ~ 2004, there is an initiation step
# that needs to be done.
# Here, we generate some random policy and calculate its feature expectation vector
# TODO: CHECK TO SEE IF GENERATING A RANDOM POLICY REPLACES THE OLD ONES OR ADDS ON TO IT
sim.gen_random_trajectory(i=10, n=1)
mu_random_current = sim.μ_estimate(trajectories=sim.random_policies, gamma=0.95)

# Now, we set our "i" counter which will help us iterate through the program
# We arbitrarily set t = a large number
i = 1
t = 1000000

# Finally, before looping, we need to init our first "guess" at w, and make mu_bar_0 = mu_random_current
# Through looping, these values will be replaced by incremental learning
mu_bar = mu_random_current
w = mu_e - mu_bar

print(sim.agents['expert'].D)
# The main algo has a break condition while t > BREAK_CONDITION (some arbitrarily small number)
# while t > 1:
#
#     # === (1a) Generate Random Policy === #
#     # This method helps to generate random trajectories. This is specifically used for the IRL algorithm
#     # NOTE: THE .build_trajectories() METHOD RESETS THE SIMULATION ENVIRONMENT WHEN DONE. NO NEED TO EXPLICITLY CALL IT
#     sim.gen_random_trajectory(i=10, n=1)
#     # IS THIS EVEN NEEDED?
#     rand_polic_state_trajs = sim.build_trajectories(trajectories=sim.random_policies)
# #
#
#     # === (1b) Estimate μ for the random policy === #
#     # Now that weh have our random policy, estimate μ for it
#     # NOTE: THIS FUNCTION TAKES AN ACTION-TRAJECTORY, NOT A STATE TRAJECTORY
#     mu_random = sim.μ_estimate(trajectories=sim.random_policies, gamma=0.95)
#
#     #  On the first iteration, we have to do some special stuff
#     if i == 1:
#
#         w = expert_feat_exp - mu_random
#
#         print(w)
#
#     t -= 100
    # === (2) Projecton Method to estimate t, w === #


    # === (3) Initiate RL algo to find optimal policy for R = w.T * ϕ === #
    # === (4) Compute μ for the learned policy from the RL algo === #
