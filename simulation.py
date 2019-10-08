"""

"""
#
# TODO: Add a self.gamma attribute so we don't need to manually feed it every call!
# expert's features and some random policy

from collections import Counter
import numpy as np
import pandas as pd

class Simulation:

    def __init__(self, agents, environment, alpha):
        self.agents = self._agent_init(agents=agents)
        self.random_policies = None
        self.simul_env = environment
        self.Q = None
        self.IRL = {}
        self.alpha = alpha

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

                # There  is excessive use of the .copy() method here. This is a silly Python thing
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
            # This is the same length as the feature vector, ϕ
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
            (list of lists) where each inner list is a list of action selections
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

            # To ensure that the output of the function is a lists of lists, we add an extra [] around the
            # return statement
            self.random_policies = [[np.random.choice(action_list, 1)[0] for _ in range(i)]]


    # The Simulation has the ability to build its own trajectories.
    # It does so by interacting with its stand-alone environment
    # This will help to build out the Q(s,a) grid!
    # Therefore, this function will return a list of str(vectors)-state-representations
    def build_trajectories(self, trajectories):
        """

        Args:
            trajectories (list of lists):

        Returns:

        """
        if not type(trajectories[0]) == list:
            raise TypeError('Trajectories must be a list of lists.')

        # Init a list that will hold all
        all_traj_holder = []

        # Looping through each given trajectory within the trajectories list
        for traj in trajectories:

            # To ensure proper results, we need to reset the simulation's environment using the
            # expert agent's action list
            self.simul_env._reset(action_list=self.agents['expert'].action_list)

            # Init an empty list
            traj_l = []

            # Looping through each state and recording it
            for action in traj:
                traj_l.append(self.simul_env._update_state(action=action, ret=True).copy())

            # Appending each trajectory to the outer holder
            all_traj_holder.append(traj_l)

        # Resetting the state
        self.simul_env._reset(action_list=self.agents['expert'].action_list)

        # Now that we have our lists of state-trajectories, return those bad-boys
        return all_traj_holder

    def irl_step(self, mu_m1, mu_bar_m2, mu_e):
        """The purpose of this function is to help compute the projection method (the so-called IRL step)

        Args:
            mu_m1: feature expectation vector from the previous step
            mu_bar_m2: avg feature expectation vector from two steps ago
            mu_e: feature expectation vector from the expert

        Returns:
            - the orthogonal projection of mu_e onto the lines through mu_bar_m2 and mu_m1
            - updated w = mu_e - mu_bar_m1
            - updated t = ||mu_e - mu_bar_m1||2
        """

        # Collection of maths explained in section 3.1 of  "Apprenticeship Learning via IRL" by Abbeel, Ng ~ 2004
        # In an effort to make it more readable (and fit syntaxically), the computation has been split
        # to multiple lines.
        # Sorry.
        numerator = (mu_m1 - mu_bar_m2).T * (mu_e - mu_bar_m2)
        denominator = (mu_m1 - mu_bar_m2).T * (mu_m1 - mu_bar_m2)
        result = numerator / denominator
        result = result * (mu_m1 - mu_bar_m2)
        mu_bar_m1 = mu_bar_m2 + result

        # Now that we have the updated value of mu_bar_m1, we can calculate the updated value of our weight's vector, w
        # After calculation, we ensure that ||w||1 <= 1
        updated_w = mu_e - mu_bar_m1
        updated_w = updated_w / np.sum(np.abs(updated_w))

        # And with this, we can calculate the L2 norm for t
        updated_t = np.linalg.norm((mu_e - mu_bar_m1), ord=2)

        return mu_bar_m1, updated_w, updated_t

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

        self.Q.loc[str(state_vector), action] = value

    def read_Q(self, state_vector, action=None):
        """

        Args:
            state_vector:
            action:

        Returns:

        """

        # This returns the entire vector of state-action pairs
        if not action:

            return self.Q.loc[str(state_vector)]

        else:

            return self.Q.loc[str(state_vector), action]

    def add_Q(self, state_vector):
        """"""

        new_state = pd.DataFrame(0,
                                 columns=self.agents['expert'].action_list,
                                 index=str(state_vector))

        self.Q = self.Q.append(new_state)

    def q_learning(self, epsilon, gamma, T, w, e=10e-3):
        """This method contains all of the mathematics necessary to converge to an optimal policy for the given
        estimated reward function, w.Tϕ

        The update step takes the following form:
        Q(S,A) <- Q(S,A) + alpha[w.Tϕ +  γ max_a Q(S',a) - Q(S,A)]

        Args:
            epsilon (float): represents the probability of choosing the greedy action at each step [0,1]
            gamma: the discount factor [0,1]
            T: the number of time steps we are allowing our agent to traverse through
            w (np.array): the vector of weights that is returned from the irl_step() method
            e: an arbitrarily small number that determines the break condition of our "convergence"

        Returns:

        """

        delta = np.inf
        previous_cumsum_r = 10000

        while delta > e:

            # Since this is a time-limited learning problem, need to run each "simulation" for a given number of steps
            # This requires init step counter to 0
            # And drawing A_0 ~ D_e
            step = 0

            # We can use np.random.choice() to draw samples from a non-uniform distribution, believe it or not
            A = np.random.choice([k for k,v in self.agents['expert'].D.items()],
                                 p=[v for k,v in self.agents['export'].D.items()])

            # Through this process, we need to keep track of the current state
            # Thankfully, I had enough forethought (read: dumb luck) to give the Simulation it's own env
            # Explicitly resetting the simulation's env
            self.simul_env._reset(action_list=self.agents['expert'].action_list)

            # IMPORTANT TO NOTE, THE Q-TABLE CONTAINS THE NON-NORMED STATE-VECTORS ϕ, THIS IS SIMPLY FOR READABILITY
            # ANY COMPUTATION WITH ϕ SHOULD BE NORMALIZED
            S = self.simul_env._update_state(action=A, ret=True)

            # Init our cumsum_reward
            cumsum_r = self.read_Q(state_vector=S, action=A)

            # As we have already chosen an action above, (this represents STEP), we will loop until
            # the step counter is == T
            while step < T:

                # Now that we are in some state, S, let's use an epsilon-greedy policy to choose our A
                # Generating random number and if it is > epsilon, choosing a random action
                # The output of this IF/ELSE block is Q(S,A) and A
                # A will help us to compute S_prime
                if np.random.random() < epsilon:

                    # Random choice of action
                    A = np.random.choice(self.agents['expert'].action_list)

                    # Getting the Q(S,A) of that A
                    q_sa = self.read_Q(state_vector=S, action=A)

                # Else, choose the greedy action in the current state
                else:

                    # Here, we do thins in reverse order
                    # First, we pull out the maximal value in the vector
                    q_sa = np.max(self.read_Q(state_vector=S))

                    # Now, we find the index of q_sa from the vector
                    # TODO: I THINK THIS ALREADY RETURNS THE COLUMN NAME IS max_ind
                    q_sa_vector = self.read_Q(state_vector=S)
                    max_ind = q_sa_vector[q_sa_vector == q_sa].index[0]

                    # TODO: SEE ABOVE
                    # Now we use this index to pull out the action from the column names in self.Q
                    A = self.Q.columns[max_ind]

                # Now that we have the action, A, that the agent will take in the step, we can
                # find S_prime
                S_prime = self.simul_env._update_state(action=A, ret=True)

                # Now that we have selected an S_prime, there needs  to be an explicit check to see if this S'
                # is within our Q-table
                if S_prime not in self.Q.index:

                    self.add_Q(state_vector=S_prime)

                # Acquiring the greedy action in the resulting state
                # This is done by pulling the entire row vector of self.Q[state_vector] and then argmax
                q_splusa = np.max(self.read_Q(state_vector=S_prime))

                # Now the only thing left to do is update dat dere Q(S,A)
                # The states, S, coming out of the environmental update are not normed, so let's do that now
                phi = S / T
                q_sa_update = q_sa + self.alpha * (np.inner(w, phi) + gamma * q_splusa - q_sa)

                # Now updating the value
                self.update_Q(state_vector=S, action=A,  value=q_sa_update)

                # Reassigning S <- S'
                S = S_prime

                # Adding to our reward
                cumsum_r += np.inner(w, phi)

                # Increment step
                step += 1

            # After we have taken all of our steps, check the total reward as compared to the last run
            # If the amount is sufficiently small, break the damn loop
            delta = np.abs(previous_cumsum_r - cumsum_r)

            previous_cumsum_r = cumsum_r

    ##################################################
    ### ==== MC LINEAR FUNCTION APPROXIMATION ==== ###
    ##################################################

    # (1) Episode generator (takes a policy, w (from  irl_step) and returns G)
    # (2) Policy evaluation (takes G, WFW)
    # (3) Policy improvement (takes list of states visited from (1)

    def mc_gradient_update(self, episode_steps, G, WFW, alpha):
        """"""

        for step in episode_steps:

            # Computing v_hat
            v_hat = np.inner(WFW, step)

            WFW = WFW + alpha * (G - v_hat) * step

        return WFW

    def mc_policy_control(self):
        """"""

        pass


    def mc_gradient(self, w, T, epsilon, alpha, e=1e-3):
        """

        Args:
            w: should already be ||w||1 <= 1
            T:
            epsilon:
            alpha:
            e:

        Returns:

        """

        # Initalizing the weight vector
        # In order to avoid confusing the reward weight vector and the value function weight vector
        # we will call the value function weight vector WFW
        WFW = [0 for x in range(len(w))]

        # Generating an episode following π
        # Begin by restarting the state
        self.simul_env._reset(action_list=self.agents['expert'].action_list)

        # Looping through T steps and choosing actions
        # For MC methods, we observe the entire episode and generate the cumulative reward
        # So, let's first generate an enitre episode

        # This will hold G
        cumsum_r = 0

        # This dictionary will help keep track of the value of the episodes
        # {[episode_number : non-normed-action-vectors]}
        episode_dict = {}

        delta = 1000000

        previous_cumsum_r = 10000000

        episode = 0

        while delta > e:

            episode_dict[episode] = []

            for step in range(T):

                # This dictionary will help keep track of Q(S,A)
                qsa_dict = {}

                # Greedy action
                if np.random.random() < epsilon:

                    # Choosing the action that gives us the highest reward
                    # For this we need to explicitly loop through each action and calculate the reward
                    for action in self.agents['expert'].action_list:

                        # Choosing  the action results in S'
                        #  .current_state = {state1: cnt1, state2, cnt2}
                        current_state = self.simul_env.current_state

                        # Transforming  the current state into a vector and norming it
                        current_state = [v for k,v in current_state.items()] / T

                        # Because our State Vectors (returned in the last command) are simply a sum of the normed actions
                        # that have been taken, we do not need to alter the actual environment
                        # instead, we can spoof S' simply by adding the norm of the proposed action to the current state
                        normed_action = action / T
                        s_prime = current_state + normed_action

                        # Calculating the value of action in the current state
                        qsa_dict[action] = np.inner(WFW, s_prime)

                    # Now that we have the value of each action, find the maximum
                    # This should generalize to the case in which several or all  actions return the same  value
                    max_action = None
                    max_value = -10000000
                    for k,v in qsa_dict.items():

                        if v > max_value:

                            max_action = k
                            max_value = v

                        else:

                            continue

                    # Now that we know what action is the greedy action, log it into episode_dict
                    episode_dict[episode].append(max_action)

                    # Now that we know our action, lets calculate the reward
                    # There is some double-computation going on here, how can we simplify this?
                    phi = (self.simul_env.current_state / T) + (max_action / T)
                    cumsum_r += np.inner(w, phi)

                    # Updating our state
                    # No need to return
                    self.simul_env._update_state(action=max_action)

                else:

                    # Random choice selection
                    A = np.random.choice(self.agents['expert'].action_list)

                    # Now that we know what action we have randomly selected, log it into episode_dict
                    episode_dict[episode].append(A)

                    # Calculating the reward
                    phi = (self.simul_env.current_state / T) + (A / T)
                    cumsum_r += np.inner(w, phi)

                    # Updating our state
                    # No need to return
                    self.simul_env._update_state(action=A)

            # Now that we have generated an episode according to e-greedy, let's perform a weight update
            WFW =

            episode += 1

            # delta = np.abs(previous_cumsum_r - cumsum_r)
            #
            # previous_cumsum_r =  cumsum_r
