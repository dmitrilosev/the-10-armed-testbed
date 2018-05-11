import numpy as np
import matplotlib.pyplot as plt
from statistics import mean


class Bandit:
	def __init__(self, number_of_actions):
		self.number_of_actions = number_of_actions
		self.action_values = np.random.normal(0, 1, number_of_actions)

	def step(self, action):
		reward = np.random.normal(self.action_values[action], 1)
		return reward


class ActionValue:
	def __init__(self, number_of_actions):
		self.number_of_actions = number_of_actions
		self.action_values = np.zeros(number_of_actions)
		self.episode = 0

	def epsilon_greedy_action(self, epsilon):
		return np.random.randint(self.number_of_actions) if np.random.uniform(0,1) < epsilon else np.argmax(self.action_values)

	def train(self, action, reward, alpha):
		self.episode += 1
		self.action_values[action] = self.action_values[action] + alpha * (reward - self.action_values[action])


def run_bandit_testbed(number_of_actions, number_of_bandits, number_of_steps, epsilon, alpha):
	bandits = [Bandit(number_of_actions) for _ in range(number_of_bandits)]
	action_values = [ActionValue(number_of_actions) for _ in range(number_of_bandits)]

	steps = []
	for step in range(number_of_steps):
		rewards = []
		for action_value, bandit in zip(action_values, bandits):
			if epsilon == -1:
				action = action_value.epsilon_greedy_action(max(0.1, (100 - len(rewards)) / 100))
			else:
				action = action_value.epsilon_greedy_action(epsilon)
			reward = bandit.step(action)
			action_value.train(action, reward, alpha)
			rewards.append(reward)
		steps.append(mean(rewards))

	return steps


steps0 = run_bandit_testbed(10, 500, 1000, 0.01, 0.1)
steps1 = run_bandit_testbed(10, 500, 1000, 0.1, 0.1)

plt.plot(range(len(steps0)), steps0, 'r')
plt.plot(range(len(steps1)), steps1, 'b')
plt.axis([0, len(steps0), min(steps0), max(steps0)])
plt.show()

