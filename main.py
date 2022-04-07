import gym
import numpy as np
from SECTIONs import MDP_VIA as via
from SECTIONs import Policy_Iteration_Algorithm as pia
from SECTIONs import Temp_diff_state_action_algo as tsa
from SECTIONs import Qlearn_without_eligibly as qwo
from SECTIONs import Qlearn_with_eligibly as qw


def finMDP(viao, env):
    print("starting position:\n")
    env.render()  # print starting position
    # Presentation of the expected rewards transition matrix and obtaining the Transition probabilities matrix.
    P_pi, R_pi, mm = viao.Intro_MDP_Agent()
    print(f"\nTransition matrix = \n\n {P_pi}\n")
    print(f"Reward vector = \n\n {R_pi}\n")
    # print("Transition probabilities and expected rewards = \n")
    # for i in mm:
    #     print(i)


def value_iter(viao, env, n):
    e = 0
    V, action, pi = viao.value_iteration()
    for i_episode in range(n):
        c = env.reset()
        for t in range(n):
            c, reward, done, info = env.step(action[c])
            if done:
                if reward == 1:
                    e += 1
                break
    print(f" agent succeeded to reach goal {e + 1} out of {n} Episodes\n")
    print('policy:\n')
    a2w = {0: 'West(<)', 1: 'south(v)', 2: 'East(>)', 3: 'North(^)'}
    policy_arrows = [a2w[x] for x in np.argmax(pi, axis=-1)]
    print(np.array(policy_arrows).reshape([-1, 4]))
    print(" ")
    print("value func vector:\n")
    print(V.reshape([-1, 4]))
    env.close()


def Policy_Iter_Algo(piao, env):
    V, pi = piao.policy_iter()
    a2w = {0: 'West(<)', 1: 'South(v)', 2: 'East(>)', 3: 'North(^)'}
    policy_arrows = np.array([a2w[x] for x in pi])
    print('*policy:\n')
    print(np.array(policy_arrows).reshape([-1, 4]))
    print('value func vector (*policy):\n')
    print(V.reshape([4, -1]))
    env.close()


def Temp_diff_algo(env, nb_states, nb_actions, nb_episodes, alpha, gamma):
    # intial value function
    v = np.zeros(16)
    counter = 0
    pi = np.ones([nb_states, nb_actions]) / nb_actions
    # loop over num of episodes
    for i in range(nb_episodes):
        done = False
        observation = env.reset()
        counter += 1
        while not done:
            # sample an action from the prob distribution of the observation
            action = np.random.choice(4, 1, p=pi[observation])
            s_, reward, done, info = env.step(action[0])
            v[observation] = v[observation] + alpha * (reward + gamma * v[s_] - v[observation])
            observation = s_
    # print("\npoli:\n")
    # print(pi)
    print("\nvalue func\n")
    print(v.reshape(4, -1))
    env.close()


def Temp_diff_st_ac(tsao, env):
    tsao.sarsa()
    env.close()


def Qlearn_no_eligibly(qwoo, env):
    qwoo.Q_learning_without_eligibly_traces()
    env.close()


def Qlearn_eligibly(qwob, env):
    qwob.Q_learning_with_eligibly_traces()
    env.close()


def main():

    np.set_printoptions(linewidth=200)  # nice printing of large arrays
    env = gym.make('FrozenLake', desc=["SFFF", "FHFH", "FFFH", "HFFG"],
                   map_name="4x4", is_slippery=True)  # Initialise variables used through script
    # env = gym.make('FrozenLake', desc=["FSFH", "FFHG", "FFHF", "FFFF"],
    #                map_name="4x4", is_slippery=False)  # Initialise variables used through script
    # env = gym.make('FrozenLake', desc=["FSFH", "FFHG", "FFHF", "GFFF"],
    #                map_name="4x4", is_slippery=True)  # Initialise variables used through script
    # # env = gym.make('FrozenLake', desc=["SFFH", "FFHF", "FHFF", "FHFG"],
    #                map_name="4x4", is_slippery=True)  # Initialise variables used through script
    # env = gym.make('FrozenLake', desc=["SFFFGF", "FHFHHF", "FHFFHF", "FHHFFF", "FFGHFH", "HFFFFF"],
    #               map_name="6x6", is_slippery=True)  # Initialise variables used through script
    # env = gym.make('FrozenLake', desc=["SFFFFF", "FHFHHF", "FHFFHF", "FHHFFF", "FFFHFH", "HFFGFF"],
    #                map_name="6x6", is_slippery=True)  # Initialise variables used through script
    env.reset()
    model = env.env.P
    nb_states = 16  # number of possible states
    nb_actions = 4  # number of actions from each state
    n = 10000
    nb_episodes = 3000
    STEPS = 200
    gamma = 0.9
    theta = 0.000001
    alpha = 0.05
    epsilon = 0.05

    viao = via(model, nb_states, nb_actions, gamma, theta)
    finMDP(viao, env)

    value_iter(viao, env, n)

    piao = pia(model, nb_states, nb_actions, gamma, theta)
    Policy_Iter_Algo(piao, env)

    Temp_diff_algo(env, nb_states, nb_actions, nb_episodes, alpha, gamma)

    tsao = tsa(alpha, gamma, epsilon, env, nb_episodes)
    Temp_diff_st_ac(tsao, env)

    qwoo = qwo(alpha, gamma, epsilon, env, nb_episodes, STEPS)
    Qlearn_no_eligibly(qwoo, env)

    qwob = qw(alpha, gamma, epsilon, env, nb_episodes, STEPS, nb_states, nb_actions)
    Qlearn_eligibly(qwob, env)


if __name__ == '__main__':
    main()
