import numpy as np
import pandas as pd
import os
from agents import Agent
from state import Environment
from simulation import Simulation

#############################
# == Information Keepers == #
#############################
# For post-analysis
run = 1
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
          action_list=['l', 'r', 'u', 'd'],
          environment=env,
          trajectories=[['r', 'u', 'r', 'u', 'r', 'u', 'r', 'u', 'r', 'u']])

# Build said expert trajectories
a.build_trajectories()

# Build the Agent's initial state distribution
a.build_D()

# Init a standalone environment for the state itself
simul_env = Environment()

# Init the simulation
sim = Simulation(agents=a, environment=simul_env, alpha=.2)

# This method will initalize a matrix of state-action pairs and their values (currently set to init all
# at 0). This  will build a matrix that represents all of the states that the expert agent has visited
sim.reset_q(trajectories=sim.agents['expert'].state_trajectories)

# Computing the feature expectation of the expert
mu_e = sim.μ_estimate(trajectories=sim.agents['expert'].trajectories, gamma=0.99)

# Generation of a random policy
# This policy gets stored into sim.random_policies, which is a list of trajectory lists
sim.gen_random_trajectory(i=10)
pi_historical.append(sim.random_policies[0])

# # Getting the feature expectation of the randomly generated policy
mu_r = sim.μ_estimate(trajectories=sim.random_policies, gamma=0.99)
mu_historical.append(mu_r)

# Iteration helper
# How to set ep_break appropriately?
i = 1
ep_break = 0.0001

# On the first iteration, some special things need to happen
# (1) Initalization of the weight vector to mu_e - mu_0
#   (i) we must ensure that ||w||1 <= 1
w = mu_e - mu_r

# w = sim.sigmoid(w)
# if not np.linalg.norm(w, 1) <= 1:
#     w = w / np.linalg.norm(w, 1)

# L2 Norm of the weights
w = w / np.linalg.norm(w, 2)

w_historical.append(w)

# (2) mu_hat = mu_r
mu_bar = mu_r.copy()
mu_bar_historical.append(mu_bar)

# First value of t
t = np.linalg.norm((mu_e - mu_bar), 2)
t_historical.append(t)


#
# Now that we have our first randomly estimate weight vector, we can proceed to the first RL step
cumsum_r_ts = sim.q_learning(break_condition=.00001,
                             num_steps=10,
                             epsilon=.25,
                             w=w,
                             gamma=0.99,
                             IRL=i)
r_historical.append(cumsum_r_ts)

# Creating our new estimate of our policy
# We do this by first computing a greedy policy
greedy_traj = sim.greedy_policy(i=10)
pi_historical.append(greedy_traj)

# Then, getting the feature expectation of the greedy traj
mu_r = sim.μ_estimate(trajectories=greedy_traj, gamma=0.99)
mu_historical.append(mu_r)

# Increment. Remember!
i += 1

while t > ep_break:

    # Taking the IRL step
    mu_bar_m1, w, t, = sim.irl_step(mu_m1=mu_historical[i-1],
                                    mu_bar_m2=mu_bar_historical[i-2],
                                    mu_e=mu_e)

    # The above IRL step will terminate with w == 'exact' when the mu_m1 == mu_e
    # If so, the .irl_step() method above will return w as 'exact', or STR
    if type(w) == str:

        print('-------------------------------')
        print('Algorithm found exact match')

        # Making a folder
        os.mkdir(f'{i}')

        # Writing the current Q-table to csv
        sim.Q.to_csv(f'{i}/q_table.csv')

        # Saving the state-mapping-thing
        pd.DataFrame(sim.state_q_mapping, index=sim.agents['expert'].action_list).to_csv(f'{i}/state_mapping.csv')

        # Writing t to a csv
        pd.DataFrame({'t': t}, index=[0]).to_csv(f'{i}/t.csv')

        # Writing the weight
        pd.DataFrame({'w': w}, index=sim.agents['expert'].action_list).to_csv(f'{i}/w.csv')

        # Writing the final policy
        pd.DataFrame({'pi': greedy_traj[0]}).to_csv(f'{i}/pi.csv')
        print('-------------------------------')

    # Appending the newly calculated information to their respective information holders
    mu_bar_historical.append(mu_bar_m1)
    w_historical.append(w)

    # RL Step
    # This method will initalize a matrix of state-action pairs and their values (currently set to init all
    # at 0). This  will build a matrix that represents all of the states that the expert agent has visited
    # sim.reset_q(trajectories=sim.agents['expert'].state_trajectories) # TODO: DON'T DO THIS?
    cumsum_r_ts = sim.q_learning(break_condition=.00001,
                             num_steps=10,
                             epsilon=.25,
                             w=w,
                             gamma=0.99,
                             IRL=i)
    r_historical.append(cumsum_r_ts)

    # New greedy policy
    greedy_traj = sim.greedy_policy(i=10)
    pi_historical.append(greedy_traj)
    mu_r = sim.μ_estimate(trajectories=greedy_traj, gamma=0.99)
    mu_historical.append(mu_r)

    i += 1
    print(t)
    if i % 30 == 0:
        print('-------------------------------')
        print(f'Iteration {i}. Saving snapshot...')

        # Making a folder
        os.mkdir(f'{i}')

        # Writing the current Q-table to csv
        sim.Q.to_csv(f'{i}/q_table.csv')

        # Saving the state-mapping-thing
        pd.DataFrame(sim.state_q_mapping, index=sim.agents['expert'].action_list).to_csv(f'{i}/state_mapping.csv')

        # Writing t to a csv
        pd.DataFrame({'t': t}, index=[0]).to_csv(f'{i}/t.csv')

        # Writing the weight
        pd.DataFrame({'w': w}, index=sim.agents['expert'].action_list).to_csv(f'{i}/w.csv')

        # Writing the final policy
        pd.DataFrame({'pi': greedy_traj[0]}).to_csv(f'{i}/pi.csv')
        print('-------------------------------')


print('-------------------------------')
print(f'Algorithm terminated at t: {t}. Saving results...')

# Making a folder
os.mkdir(f'{i}')

# Writing the current Q-table to csv
sim.Q.to_csv(f'{i}/q_table.csv')

# Saving the state-mapping-thing
pd.DataFrame(sim.state_q_mapping, index=sim.agents['expert'].action_list).to_csv(f'{i}/state_mapping.csv')

# Writing t to a csv
pd.DataFrame({'t': [t]}, index=[0]).to_csv(f'{i}/t.csv')

# Writing the weight
pd.DataFrame({'w': w}, index=sim.agents['expert'].action_list).to_csv(f'{i}/w.csv')

# Writing the final policy
pd.DataFrame({'pi': greedy_traj[0]}).to_csv(f'{i}/pi.csv')
print('-------------------------------')

#     print(f't: {t}')
#     print(f'weight: {w}')
#     print(f'pi: {greedy_traj}')
#     print('---------------------------------------------------')
#     if i % 50 == 0:
#         print(w_historical[len(w_historical)-1])
#         print(pi_historical[len(pi_historical)-1])
#
#         print('-------------')
#         # a, ratio = sim.compare_e_a(w=w,
#         #                            a_trajectory=pi_historical[len(pi_historical) - 1][0],
#         #                            num_steps=10)
#         #
#         # print(f"Ratio: {ratio}")
#
# w_historical.append(w)
# print(w_historical)
# print(pi_historical)
# print(f't: {t}')
# print('-------------')
# # a, ratio = sim.compare_e_a(w=w,
# #                            a_trajectory=pi_historical[len(pi_historical) - 1][0],
# #                            num_steps=10)
# # print(a)
# #
# # print(f"Ratio: {ratio}")