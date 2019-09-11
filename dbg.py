from agents import Agent
from simulation import Simulation
from state import Environment

env = Environment()

a = Agent(type='expert',
          action_list=['a', 'b', 'c', 'd', 'e', 'f', 'g'],
          environment=env,
          trajectories=[['a', 'b', 'c', 'e', 'g', 'b', 'c', 'e', 'g',],
                        ['a', 'b', 'c', 'a', 'g', 'g', 'a', 'g', 'g'],
                        ['c', 'd', 'f', 'b', 'c', 'a', 'd', 'f', 'b', 'c']])

a.build_trajectories()

a.build_D()

simul_env = Environment()

sim = Simulation(agents=a, environment=simul_env)
sim.gen_random_trajectory(i=22, n=2)

fv = sim.μ_estimate(trajectories=sim.random_policies, gamma=0.5)
print(fv)

#
# print(env.current_state)


# from cvxopt import matrix
# from cvxopt import solvers #convex optimization library
# import numpy as np
#

# print(matrix(2 * np.eye(1), tc='d'))
#
# def optimization(self):  # implement the convex optimization, posed as an SVM problem
#     m = len(self.expertPolicy)
#     P = matrix(2.0 * np.eye(m), tc='d')  # min ||w|| IDENTITY MATRIX?
#     q = matrix(np.zeros(m), tc='d')
#     policyList = [self.expertPolicy]
#     h_list = [1]
#     for i in self.policiesFE.keys():
#         policyList.append(self.policiesFE[i])
#         h_list.append(1)
#     policyMat = np.matrix(policyList)
#     policyMat[0] = -1 * policyMat[0]
#     G = matrix(policyMat, tc='d')
#     h = matrix(-np.array(h_list), tc='d')
#
#     # Min (1/2)x.T * Px + q.T * x
#     # subject to Gx <= h, Ax = b
#     # AND
#     # Max
#     sol = solvers.qp(P, q, G, h)
#
#     weights = np.squeeze(np.asarray(sol['x']))
#     norm = np.linalg.norm(weights)
#     weights = weights / norm
#     return weights  # return the normalized weights