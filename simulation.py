"""

"""
#
# TODO: Calculate the feature expectations of a given policy -- we'll need todo this for the
# expert's features and some random policy

from collections import Counter
import numpy as np
import pandas as pd

class Simulation:

    def __init__(self, agents, environment):
        self.agents = self._agent_init(agents=agents)
        self.random_policies = None
        self.simul_env = environment
        self.Q = None

    def _agent_init(self, agents):
        """

        Args:
            agents:

        Returns:

        """

        if not type(agents) == list:
            agents = [agents]

        # First collecting a count of agent types
        count_dict = Counter([agent.type for agent in agents])

        # Subsetting the count_dict to a dict that contains duplicate Agent.type
        subset = {k: v for k, v in count_dict.items() if v > 1}

        if len(subset) > 0:
            raise AttributeError(f'You may only have one type of each agent. You have: \n {subset}')

        # If there are no duplicate types, return a dict that will help to reference
        return {agent.type: agent for agent in agents}

    def μ_estimate(self, trajectories, gamma):
        """"""

        if not 0 <= gamma <= 1:
            raise ValueError('Gamma ∈ [0,1]')

        if not type(trajectories) == list:
            trajectories = [trajectories]

        if not type(gamma) == float:
            raise TypeError('Gamma must be a float.')

        # Init a list that will hold the Feature Expectations of each trajectory
        outer_traj = []

        # Due to poor design choices, (or are they actually genius?), we will need to draw from our
        # simulation environment if we are feeding this function randomly generated trajectories
        # Ultimately, I have copied over the procedure from the Agent() class.
        # Oops.
        if not type(trajectories[0]) == dict:

            # Init a list that will hold all
            all_traj_holder = []

            # Looping through each given trajectory within the trajectories list
            for trajectory in trajectories:

                # If the user does wish to do so, hit the reset button on the environment
                self.simul_env._reset(action_list=self.agents['expert'].action_list)

                # Init an empty list
                traj_l = []

                # Appending in our S_0
                # There  is excessive use of the .copy() method here. This is a silly Python thing

                # I have commented out the S0 as being  a vector of 0s!
                # Seems unnecessary to have and complicates the Mu calc? 9.10.2019
                # traj_l.append(self.environment.current_state.copy())

                # Looping through each state and recording it
                for action in trajectory:
                    traj_l.append(self.simul_env._update_state(action=action, ret=True).copy())

                # Appending each trajectory to the outer holder
                all_traj_holder.append(traj_l)

            # Setting the trajectories for the Agent
            trajectories = all_traj_holder

            # Resetting the environment back to its initial state
            self.simul_env._reset(action_list=self.agents['expert'].action_list)

        # Looping through the list of trajectories...
        for traj in trajectories:

            # Init an inner list holder for the
            inner_traj = []

            # Now looping through each step in the traj
            # This data should be structured like a dictionary:
            # {'a': 0, 'b': 2, 'c': 3}
            # We use the enumerate() functionality, as the i from the counter
            # will ultimately become the exponent for the discount factor, γ
            for i, step in enumerate(traj):

                # We need to ensure that the ϕ vector is in the same order
                # We can do this by looping through each of the
                phi = np.array([step[x] for x in self.agents['expert'].action_list])

                # Normalizing this [0,1]
                # This helps later on with L1 norming
                phi = phi / len(traj)

                # Multiplying ϕ by our degrading discount factor
                phi = (gamma ** i) * phi

                inner_traj.append(phi)

            # Once we have finished looping through all of the steps, let us sum the vectors
            # Let us start by init'ing a vector of zeros
            init_vector = np.zeros(len(inner_traj[0]))

            # Then, looping through each discounted state and summing them together
            for state in inner_traj:

                init_vector += state

            # Finally, appending the resulting Feature vector to the outer list
            outer_traj.append(init_vector)

        # Now summing together each of the feature expectation vectors in outer_traj
        init_vector = np.zeros(len(outer_traj[0]))

        for fv in outer_traj:

            init_vector += fv

        # AND FINALLY, averaging this to get our final Feature Expectation vector
        init_vector = init_vector / len(outer_traj)

        # Because these Feature Expectation vectors could be from randomly generated policies,
        # we will not store them into the simulation but simply return them, to the outside world
        # to be free as Free Range vectors 100% organic!
        return init_vector

    def gen_random_trajectory(self, i, n=1):
        """

        Args:
            i (int): the number of time steps to generate in the trajectories
            n (int): the number of trajectories to generate

        Returns:

        """

        if not 'expert' in [k for k, v in self.agents.items()]:
            raise AttributeError('You do not have an expert agent in your simulation.')


        if not type(i) == int:
            raise TypeError('i must be an int.')

        if not type(n) == int:
            raise TypeError('n must be an int.')

        # Pulling out the action list from the expert agent
        action_list = self.agents['expert'].action_list

        if not n == 1:

            traj_list = []

            for traj in range(n):

                traj_list.append([np.random.choice(action_list, 1)[0] for _ in range(i)])

            self.random_policies = traj_list

        else:

            self.random_policies = [np.random.choice(action_list, 1)[0] for _ in range(i)]


    # The Simulation has the ability to build its own trajectories.
    # It does so by interacting with its stand-alone environment
    # This will help to build out the Q(s,a) grid!
    # Therefore, this function will return a list of str(vectors)-state-representations
    def build_trajectories(self, trajectories):
        """"""

        # Init a list that will hold all
        all_traj_holder = []

        # Looping through each given trajectory within the trajectories list
        for traj in trajectories:

            # To ensure proper results, we need to reset the simulation's environment using the
            # expert agent's action list
            self.simul_env._reset(action_list=self.agents['expert'].action_list)

            # Init an empty list
            traj_l = []

            # Appending in our S_0
            # There  is excessive use of the .copy() method here. This is a silly Python thing

            # I have commented out the S0 as being  a vector of 0s!
            # Seems unnecessary to have and complicates the Mu calc? 9.10.2019
            # traj_l.append(self.environment.current_state.copy())

            # Looping through each state and recording it
            for action in traj:
                traj_l.append(self.simul_env._update_state(action=action, ret=True).copy())

            # Appending each trajectory to the outer holder
            all_traj_holder.append(traj_l)

        # Resetting the state
        self.simul_env._reset(action_list=self.agents['expert'].action_list)

        # Now that we have our lists of state-trajectories, return those bad-boys
        return all_traj_holder

    def reset_q(self, trajectories):
        """The purpose of this method is to build all of the unique state vectors and then use those to begin
        construction of the state-action-pair space. As a principle, this software will use "lazy computation". That is,
        we will not store every possible state-action pair in memory, as our state-action-space can be inf or near-inf.
        Instead, we will check the index of self.Q to determine if we have been to this state before. If we have not,
        we will initialize it on the fly. If we have, then great.

        HERE, WE DO NOT CARE ABOUT GAMMA, AS WE ONLY NEED THE STATE REPRESENTATION, NOT ϕ

        Args:
            trajectories (list of dicts): this function expects a list of state dictionaries returned from
                the build_trajectories() method from either the Expert Agent or the Simulation itself

        Returns:

        """

        if not type(trajectories) == list:
            trajectories = [trajectories]

        if not type(trajectories[0][0]) == dict:
            raise AttributeError('You have not provided trajectories in the proper format. Please refer to the '
                                 'documentation, thanks mkay.')

        state_vector_holder = []

        # We need to pull out all of the values from the key:value pairs from our state trajectories dictionaries.
        # These values will be normalized by the length of our trajectories
        for traj in trajectories:

            # # Init a holder list
            # traj_holder = []

            # Now, within each trajectory, there are snapshots of each state
            for state in traj:

                # WE DO NOT CARE ABOUT THE LENGTH OF THE TRAJECTORY HERE, AS THIS IS NOT ϕ
                # In addition, we need to ensure that our state-vector representations STAY IN ORDER.
                # As such, we need to loop through the expert's action values...
                state_holder = []

                for action in self.agents['expert'].action_list:

                    state_holder.append(state[action])

                # Finally, we append a str representation of a np.array() into the state_vector_holder
                state_vector_holder.append(str(np.array(state_holder)))


        # Now that we have our list of historically visited states, let's reduce this list to only the unique
        # state-vector representations...
        state_vector_holder = set(state_vector_holder)

        # And now, let's init a Pandas DF
        self.Q = pd.DataFrame(0,
                              columns=self.agents['expert'].action_list,
                              index=state_vector_holder)

    def update_Q(self, state_vector, action, value):
        """

        Args:
            state_vector:
            action:
            value:

        Returns:

        """

        self.Q.loc[state_vector, action] = value

    def read_Q(self, state_vector, action=None):
        """

        Args:
            state_vector:
            action:

        Returns:

        """

        # This returns the entire vector of state-action pairs
        if not action:

            return self.Q.loc[state_vector]

        else:

            return self.Q.loc[state_vector, action]

    def add_Q(self, state_vector):
        """"""

        new_state = pd.DataFrame(0,
                                 columns=self.agents['expert'].action_list,
                                 index=state_vector)

        self.Q = self.Q.append(new_state)
