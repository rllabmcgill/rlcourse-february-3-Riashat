import gym
import matplotlib
import numpy as np
import sys
import random
import matplotlib
matplotlib.use('TkAgg')

from collections import defaultdict

# from lib.envs.blackjack import BlackjackEnv
# from lib import plotting


# env = BlackjackEnv()

# from lib.envs.gridworld import GridworldEnv
# env = GridworldEnv()


from lib import plotting
from lib.envs.cliff_walking import CliffWalkingEnv
env = CliffWalkingEnv()




def make_epsilon_greedy_policy(Q, epsilon, nA):


	def policy_fn(observation):
		A = np.ones(nA, dtype=float) * epsilon/nA
		best_action = np.argmax(Q[observation])
		A[best_action] += (1.0 - epsilon)
		return A

	return policy_fn




def mc_control_epsilon_greedy(env, num_episodes, discount_factor=1.0, epsilon=0.1):
	   
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)

    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    

    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    stats = plotting.EpisodeStats(episode_lengths = np.zeros(num_episodes+1), episode_rewards = np.zeros(num_episodes+1))

    for i_episode in range(1, num_episodes + 1):
        # Print out which episode we're on, useful for debugging.

        # Generate an episode.
        # An episode is an array of (state, action, reward) tuples
        episode = []
        state = env.reset()

        for t in range(100):

            probs = policy(state)

            action = np.random.choice(np.arange(len(probs)), p=probs)
            next_state, reward, done, _ = env.step(action)

            episode.append((state, action, reward))

            if done:
                break

            state = next_state

       # Find all (state, action) pairs we've visited in this episode
        # We convert each state to a tuple so that we can use it as a dict key
        sa_in_episode = set([(tuple(x[0]), x[1]) for x in episode])
        for state, action in sa_in_episode:
            sa_pair = (state, action)
            # Find the first occurance of the (state, action) pair in the episode
            first_occurence_idx = next(i for i,x in enumerate(episode)
                                       if x[0] == state and x[1] == action)

            # Sum up all rewards since the first occurance
            G = sum([x[2]*(discount_factor**i) for i,x in enumerate(episode[first_occurence_idx:])])

            # Calculate average return for this state over all sampled episodes
            returns_sum[sa_pair] += G

            returns_count[sa_pair] += 1.0

            Q[state][action] = returns_sum[sa_pair] / returns_count[sa_pair]
        
        # The policy is improved implicitly by changing the Q dictionar
    
    return Q, policy


def main():
	# Q, policy = mc_control_epsilon_greedy(env, num_episodes=500000, epsilon=0.1)
	Q, policy= mc_control_epsilon_greedy(env, num_episodes=300000, epsilon=0.4)


	# For plotting: Create value function from action-value function
	# by picking the best action at each state
	V = defaultdict(float)
	for state, actions in Q.items():
	    action_value = np.max(actions)
	    V[state] = action_value


	plotting.plot_value_function(V, title="On-Policy MC Control, 30000 steps - Value Function")

if __name__ == '__main__':
    main()                	



