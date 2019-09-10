from agents import Agent
from simulation import Simulation
from state import Environment

env = Environment()

a = Agent(type='expert',
          action_list=['a', 'b', 'c'],
          environment=env,
          trajectories=[['a', 'b', 'c', 'b'],
                        ['a', 'b', 'c', 'a'],
                        ['c', 'b', 'c', 'a']])

a.build_trajectories()

a.build_D()

sim = Simulation(agents=a)

sim.μ_estimate(trajectories=sim.agents['expert'].state_trajectories, gamma=.5)

# sim.gen_random_trajectory(i=5, n=2)
#
# print(env.current_state)


# from cvxopt import matrix
# from cvxopt import solvers #convex optimization library
# import numpy as np
#
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