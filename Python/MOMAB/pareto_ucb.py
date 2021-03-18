
import numpy as np
import random as random
import math
import matplotlib.pyplot as plt

# This is an implementation of the pareto ucb algorithm outlined in Designing multi-objective multi-armed bandits algorithms: A study.
# Please cite the following paper if using this code:
# @INPROCEEDINGS{drugan2013momab,
# author={M. M. {Drugan} and A. {Nowe}},
# booktitle={The 2013 International Joint Conference on Neural Networks (IJCNN)}, 
# title={Designing multi-objective multi-armed bandits algorithms: A study}, 
# year={2013},
# volume={},
# number={},
# pages={1-8},
# doi={10.1109/IJCNN.2013.6707036}}


class Bandit:
	def __init__(self, p_true, p_false):
		self.p_true = p_true
		self.p_false = p_false
		self.n = 1
		self.return_vector = [p_true, p_false]

	def pull_arm(self):
		return [self.p_true, self.p_false]

class Pareto_ucb:
	def __init__(self, num_arms):
		self.n = num_arms
		self.count = []
		self.arm_mean = []
		self.num_pulls = 0
		self.arms = []
		self.d = 2
		init = [0.0, 0.0]
		init_ = np.array(init)

		for i in range(num_arms):
			self.count.append(0)
			self.arm_mean.append(init_)
			self.arms.append(i)

		self.pareto_set = []
		self.a_star = 6


	def update(self, action, _return_):
		self.num_pulls += 1
		self.count[action] += 1
		self.arm_mean[action] = self.arm_mean[action] + np.array(_return_)

	def select_action(self):		
		return np.random.choice(self.arms)

	def UCB(self, arm):
		sqrt = (self.d * 4) ** (1/4)
		#print("test", (self.d * 4) ** (1/4))

		top = 2 * np.log(self.num_pulls * (sqrt))
		bottom = self.count[arm]
		ucb = np.sqrt(top/bottom)

		return ucb

	def pareto_dominance(self):
		p = 0
		
		self.pareto_set = []
		for i in range(len(self.arms)):
			inSet = True
			arm = (np.array(self.arm_mean[i])/self.count[i]) + self.UCB(i)
			for j in range(len(self.arms)):
				if j == i:
					dominated = False
				else:
					arm_ = (self.arm_mean[j]/self.count[j]) + self.UCB(j)
					dominated = self.check_pareto_domination(arm_, arm)

				if dominated == True:
					inSet = False
					break

			if inSet == True:
				self.pareto_set.append(i)

		return self.pareto_set
				

	def check_pareto_domination(self, u, v):
	    dominated = False  # assume that v is not dominated by u
	    num_greater = 0
	    num_close = 0
	    num_less = 0
	    for c in range(len(u)):
	        if u[c] > v[c]:
	            num_greater += 1
	        elif u[c] < v[c]:
	            num_less += 1
	        elif math.isclose(u[0], v[0], rel_tol=1e-09, abs_tol=0.0):
	            num_close += 1
	    if num_less >= 1:
	        dominated = False
	    elif num_greater >= 1 and num_close == (len(u)-num_greater):
	        dominated = True
	    return dominated




def main():

	timestep = 100000
	num_arms = 6
	agent = Pareto_ucb(num_arms)
	bandits = []

	bandits.append(Bandit(0.55, 0.5)) 	
	bandits.append(Bandit(0.53, 0.51))
	bandits.append(Bandit(0.52, 0.54)) 
	bandits.append(Bandit(0.5, 0.57))
	bandits.append(Bandit(0.51, 0.51))
	bandits.append(Bandit(0.5, 0.5))

	returns = []
	average_utility = 0
	pareto_set = []
	

	for i in range(timestep):
		if i == 0:
			for i in range(len(bandits)):
				_return_ = bandits[i].pull_arm()
				agent.update(i, _return_)

		action = agent.select_action()
		_return_ = bandits[action].pull_arm()
		agent.update(action, _return_)
		pareto_index = agent.pareto_dominance()

	for val in pareto_index:
		pareto_set.append(bandits[val].return_vector)

	print("pareto front", pareto_set)

if __name__ == "__main__":
    main()

