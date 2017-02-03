import gym
import matplotlib
import numpy as np
import sys

from collections import defaultdict

from lib.envs.blackjack import BlackjackEnv
from lib import plotting

matplotlib.style.use('ggplot')

env = BlackjackEnv()



def Boltzmann_Behaviour_Policy(Q, softmax_prob, nA):

    A = softmax_prob

    def policy_fn(observation):   
        # best_action = np.argmax(Q[observation])
        # A[best_action] += (1.0 - softmax_prob)
        return A
    return policy_fn


def create_epsilon_greedy_policy(Q, epsilon, nA):

    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon/nA
        best_action = np.argmax(Q[observation])
        A[best_action] += ( 1.0 - epsilon)
        return A

    return policy_fn



def create_greedy_policy(Q):    
    def policy_fn(state):
        A = np.zeros_like(Q[state], dtype=float)
        best_action = np.argmax(Q[state])
        A[best_action] = 1.0
        return A
    return policy_fn







def mc_control_importance_sampling(env, num_episodes, discount_factor=1.0, epsilon=0.2):
    
    # The final action-value function.
    # A dictionary that maps state -> action values
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    # The cumulative denominator of the weighted importance sampling formula
    # (across all episodes)
    C = defaultdict(lambda: np.zeros(env.action_space.n))
    
    # Our greedily policy we want to learn
    target_policy = create_greedy_policy(Q)


    all_action_values = np.zeros(1)
    for st, action_values in Q.items():
        action_value = np.max(action_values)
        all_action_values = np.append(all_action_values, action_value)

    
    #using softmax probabilities for action selection
    softmax_prob = np.exp(all_action_values)/ np.sum(np.exp(all_action_values), axis=0)     

    behavior_policy = Boltzmann_Behaviour_Policy(Q, softmax_prob, env.action_space.n)
        
    for i_episode in range(1, num_episodes + 1):

        # Generate an episode.
        # An episode is an array of (state, action, reward) tuples
        episode = []
        state = env.reset()
        for t in range(100):
            # Sample an action from our policy
            probs = behavior_policy(state)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state
        
        # Sum of discounted returns
        G = 0.0
        # The importance sampling ratio (the weights of the returns)
        W = 1.0
        # For each step in the episode, backwards
        for t in range(len(episode))[::-1]:
            state, action, reward = episode[t]

            # Update the total reward since step t
            G = discount_factor * G + reward

            # Update weighted importance sampling formula denominator
            C[state][action] += W
            # Update the action-value function using the incremental update formula (5.7)
            # This also improves our target policy which holds a reference to Q
            Q[state][action] += (W / C[state][action]) * (G - Q[state][action])
            # If the action taken by the behavior policy is not the action 
            # taken by the target policy the probability will be 0 and we can break
            if action !=  np.argmax(target_policy(state)):
                break
            W = W * 1./behavior_policy(state)[action]
        
    return Q, behavior_policy



def main():


    Q, behaviour_policy = mc_control_importance_sampling(env, num_episodes=50000)

    V = defaultdict(float)
    for state, action_values in Q.items():
        action_value = np.max(action_values)
        V[state] = action_value

    plotting.plot_value_function(V, title="Optimal Value Function")


if __name__ == '__main__':
    main()                  


        