import numpy as np
import matplotlib.pyplot as plt
import random


class Qlearn_with_eligibly():

    def __init__(self,  alpha, gamma, epsilon, env, nb_episodes, STEPS, nb_states, nb_actions):
        self.env = env
        self.q = np.ones((16, 4))
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.nb_episodes = nb_episodes
        self.STEPS = STEPS
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.et_table = np.zeros((self.nb_states, self.nb_actions))
        self.learning_rate = 0.1
        self.discount_rate = 0.9
        self.exploration_rate = 1
        self.max_exploration_rate = 1
        self.min_exploration_rate = 0.01
        self.exploration_decay_rate = 0.001

    def average_performance(self):
        acc_returns = 0.
        n = 500
        for i in range(n):
            done = False
            s = self.env.reset()
            while not done:
                a = np.argmax(self.q[s])
                s, reward, done, info = self.env.step(a)
                acc_returns += reward
        return acc_returns / n

    def Q_learning_with_eligibly_traces(self):
        q_performance = np.ndarray(self.nb_episodes // self.STEPS)
        # Q learning algorithem
        for episode in range(int(self.nb_episodes)):
            state = self.env.reset()
            # done = False
            # reward_current_episode = 0

            for steps in range(int(self.STEPS)):
                # Exploration-Explotation trade-off
                exploration_rate_thresh = random.uniform(0, 1)
                if exploration_rate_thresh > self.exploration_rate:
                    action = np.argmax(self.q[state, :])
                else:
                    action = self.env.action_space.sample()

                new_state, reward, done, info = self.env.step(action)

                # Update Q-table and Eligibility table
                delta = reward + self.discount_rate * np.max(self.q[new_state, :]) - self.q[state, action]
                self.et_table[state, action] = self.et_table[state, action] + 1

                for update_state in range(int(self.nb_states)):
                    for update_action in range(int(self.nb_actions)):
                        self.q[update_state, update_action] = self.q[update_state, update_action] \
                                                              + self.learning_rate * delta * \
                                                               self.et_table[update_state, update_action]
                        self.et_table[update_state, update_action] = self.discount_rate * self.gamma \
                                                                     * self.et_table[update_state, update_action]
                state = new_state

                if episode % self.STEPS == 0:
                    q_performance[episode // self.STEPS] = self.average_performance()
                if done == True:
                    break

            # Exploration rate decay
            self.exploration_rate = self.min_exploration_rate + (self.max_exploration_rate - self.min_exploration_rate)\
                               * np.exp(-self.exploration_decay_rate * episode)

        plt.figure(figsize=(25, 8))
        plt.plot(self.STEPS * np.arange(self.nb_episodes // self.STEPS), q_performance, color='black')
        plt.xlabel("epochs")
        plt.ylabel("average reward of an epoch")
        plt.title("Learning progress for Q-Learning with eligibly traces")

        # Print Q-Table
        print("\n\n******* Q-Table *******\n")
        print(self.q)

        print("\n\n******* ET-Table *******\n")
        print(self.et_table)
        # print(len(q_performance))
        plt.show()


class Qlearn_without_eligibly():

    def __init__(self,  alpha, gamma, epsilon, env, nb_episodes, STEPS):
        self.env = env
        self.q = np.ones((16, 4))
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.nb_episodes = nb_episodes
        self.STEPS = STEPS

    def action_epsilon_greedy(self, s):
        if np.random.rand() > self.epsilon:
            return np.argmax(self.q[s])
        return np.random.randint(4)

    def average_performance(self):
        acc_returns = 0.
        n = 500
        for i in range(n):
            done = False
            s = self.env.reset()
            while not done:
                a = np.argmax(self.q[s])
                s, reward, done, info = self.env.step(a)
                acc_returns += reward
        return acc_returns / n

    def Q_learning_without_eligibly_traces(self):
        q_performance = np.ndarray(self.nb_episodes // self.STEPS)
        # Q-Learning: Off-policy TD control algorithm

        for i in range(int(self.nb_episodes)):
            done = False
            s = self.env.reset()
            while not done:
                a = self.action_epsilon_greedy(s)  # behaviour policy
                new_s, reward, done, info = self.env.step(a)
                a_max = np.argmax(self.q[new_s])  # estimation policy
                self.q[s, a] = self.q[s, a] + self.alpha * (reward + self.gamma * self.q[new_s, a_max] - self.q[s, a])
                s = new_s

            # for plotting the performance
            if i % self.STEPS == 0:
                q_performance[i // self.STEPS] = self.average_performance()
        #plt.figure(figsize=(25, 8))
        plt.plot(self.STEPS * np.arange(self.nb_episodes // self.STEPS), q_performance)
        plt.xlabel("epochs")
        plt.ylabel("average reward of an epoch")
        #plt.title("Learning progress for Q-Learning without eligibly traces")
        # print(len(q_performance))
        #plt.show()


class Temp_diff_state_action_algo():

    def __init__(self,  alpha, gamma, epsilon, env, nb_episodes):
        self.env = env
        self.q = None
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.nb_episodes = nb_episodes

    def action_epsilon_greedy(self, s):
        if np.random.rand() > self.epsilon:
            return np.argmax(self.q[s])
        return np.random.randint(4)

    # SARSA: On-policy TD control algorithm
    def sarsa(self):
        counter = 0
        cc = 0
        if self.q is None:
            self.q = np.zeros((16, 4))
            print("\n\n******* Initial Q-Table *******\n")
            print(self.q)
        for i in range(int(self.nb_episodes)):
            done = False
            s = self.env.reset()
            cc +=1
            while not done:
                counter += 1
                a = self.action_epsilon_greedy(s)
                new_s, reward, done, _ = self.env.step(a)
                new_a = self.action_epsilon_greedy(new_s)
                self.q[s, a] = self.q[s, a] + self.alpha * (reward + self.gamma * self.q[new_s, new_a] - self.q[s, a])
                s = new_s
                a = new_a

        print("\nfinished with", counter, "Steps for ", cc, "episodes \n")
        print("\n\n******* Final Q-Table *******\n")
        print(self.q)
        return self.q  # , progress


class Policy_Iteration_Algorithm():

    def __init__(self, model, nb_states, nb_actions, gamma, theta):
        self.model = model
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.gamma = gamma
        self.theta = theta

    def policy_iter(self):
        """Policy Iteration Algorithm
        Params:
            env - environment with following required memebers:
                nb_states - number of states
                nb_action - number of actions
                model     - prob-transitions and rewards for all states and actions, see note #1
            gamma (float) - discount factor
            theta (float) - termination condition
        """
        # 1. Initialization
        V = np.zeros(self.nb_states)
        pi = np.zeros(self.nb_states, dtype=int)  # greedy, always pick action 0
        counter = 0
        while True:
            # 2. Policy Evaluation
            while True:
                counter += 1
                delta = 0
                for s in range(self.nb_states):
                    v = V[s]
                    V[s] = self.sum_sr(V=V, s=s, a=pi[s])
                    delta = max(delta, abs(v - V[s]))
                if delta < self.theta:
                    break

            # 3. Policy Improvement
            policy_stable = True
            for s in range(self.nb_states):
                old_action = pi[s]
                pi[s] = np.argmax([self.sum_sr(V=V, s=s, a=a)  # list comprehension
                                   for a in range(self.nb_actions)])
                if old_action != pi[s]:
                    policy_stable = False
            if policy_stable:
                print("\nentered stop criteria after", counter, "iterations with delta"
                                                                "< theta :", delta, " < ", self.theta, "\n")
                break
        return V, pi

    def sum_sr(self, V, s, a):
        """Calc state-action value for state 's' and action 'a'"""
        tmp = 0  # state value for state s
        for p, s_, r, _ in self.model[s][a]:  # see note #1 !
            # p  - transition probability from (s,a) to (s')
            # s_ - next state (s')
            # r  - reward on transition from (s,a) to (s')
            tmp += p * (r + self.gamma * V[s_])
        return tmp


class MDP_VIA():

    def __init__(self, model, nb_states, nb_actions, gamma, theta):
        self.model = model
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.gamma = gamma
        self.theta = theta

    def Intro_MDP_Agent(self):
        P_pi = np.zeros([self.nb_states, self.nb_states])  # transition probability matrix (s) to (s')
        R_pi = np.zeros([self.nb_states])
        policy = np.ones([self.nb_states, self.nb_actions]) / self.nb_actions  # 0.25 probability for each action
        mm = []

        for s in range(self.nb_states):
            for a in range(self.nb_actions):
                mm.append(self.model[s][a])
                for p_, s_, r_, _ in self.model[s][a]:
                    # p_ - transition probability from (s,a) to (s')
                    # s_ - next state (s')
                    # r_ - reward on transition from (s,a) to (s')
                    P_pi[s, s_] += policy[s, a] * p_  # transition probability (s) -> (s')
                    Rsa = p_ * r_  # exp. reward from (s,a) to any next state
                    R_pi[s] += policy[s, a] * Rsa  # exp. reward from (s) to any next state
        assert np.alltrue(np.sum(P_pi, axis=-1) == np.ones([self.nb_states]))  # rows should sum to 1
        # print(policy)
        return P_pi, R_pi, mm

    def argmax(self, V, pi, action, s):
        e = np.zeros(self.nb_actions)
        vv = []
        for a in range(self.nb_actions):  # iterate for every action possible
            q = 0
            P = np.array(self.model[s][a])
            (x, y) = np.shape(P)  # for Bellman Equation

            for i in range(x):  # iterate for every possible states
                s_ = int(P[i][1])  # S' - Sprime - possible succesor states
                p = P[i][0]  # Transition Probability P(s'|s,a)
                r = P[i][2]  # Reward

                q += p * (r + self.gamma * V[s_])  # calculate action_ value q(s|a)
                e[a] = q
            vv.append(e[a])
        m = np.argmax(e)
        action[s] = m  # Take index which has maximum value
        pi[s][m] = 1  # update pi(a|s)

        return pi, vv

    def bellman_optimality_update(self, V, s):  # update the stae_value V[s] by taking
        pi = np.zeros((self.nb_states, self.nb_actions))  # action which maximizes current value
        e = np.zeros(self.nb_actions)
        count = 0  # STEP1: Find
        for a in range(self.nb_actions):
            q = 0  # iterate for all possible action
            P = np.array(self.model[s][a])
            (x, y) = np.shape(P)

            for i in range(x):
                s_ = int(P[i][1])
                p = P[i][0]
                r = P[i][2]
                q += p * (r + self.gamma * V[s_])
                e[a] = q
                count += 1

        m = np.argmax(e)
        pi[s][m] = 1
        value = 0
        for a in range(self.nb_actions):
            u = 0
            P = np.array(self.model[s][a])
            (x, y) = np.shape(P)

            for i in range(x):
                s_ = int(P[i][1])
                p = P[i][0]
                r = P[i][2]
                u += p * (r + self.gamma * V[s_])
            value += pi[s, a] * u
        # print(value)
        V[s] = value
        return V[s]

    def value_iteration(self):
        V = np.zeros(self.nb_states)  # initialize v(0) to arbitory value, my case "zeros"
        counter = 0
        while True:
            delta = 0
            counter += 1
            for s in range(self.nb_states):  # iterate for all states
                v = V[s]
                self.bellman_optimality_update(V, s)  # update state_value with bellman_optimality_update
                delta = max(delta, abs(v - V[s]))  # assign the change in value per iteration to delta
            if delta < self.theta:
                print("\n Stop conditions:")
                print(f"delta < theta : delta = {delta}\n")
                break  # if change gets to negligible
                # --> converged to optimal value
        pi = np.ones([self.nb_states, self.nb_actions]) / self.nb_actions
        action = np.zeros((self.nb_states))
        # policy = np.ones([env.env.nS, env.env.nA]) / env.env.nA  # 0.25 probability for each action

        print("value matrix:")
        print(" ")
        for s in range(self.nb_states):
            pi, vv = self.argmax(V, pi, action, s)  # extract optimal policy using action value
            print(vv)
        print(" ")
        print("value iteration for ", counter, " iterations")
        return V, action, pi  # optimal value funtion, optimal policy

