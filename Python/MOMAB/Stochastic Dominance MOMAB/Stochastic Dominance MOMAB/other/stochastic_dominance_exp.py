
import numpy as np
import random as random
import matplotlib.pyplot as plt

class Bandit:
	def __init__(self, p_true, p_false):
		self.p_true = p_true
		self.p_false = p_false

	def pull_arm(self):
		val = random.random()

		if val <= self.p_true:
			return 1
		else:
			return 0

class Sampler_exp:
	def __init__(self, num_arms):
		self.arms = []
		for i in range(num_arms):
			self.arms.append([1])

	
	def update(self, _return_, arm):
		self.arms[arm].append(_return_)

	def getDist(self, arm):
		return self.arms[arm]

	def select_arm(self):
		values = []
		for i in range(len(self.arms)):
			values.append(np.random.choice(self.arms[i]))
		
		arm_to_pull = np.argmax(values)

		return arm_to_pull

class Sampler:
	def __init__(self):
		self.arm = []

	
	def update(self, _return_):
		self.arm[arm].append(_return_)

	def getDist(self):
		return self.arm

	#def select_arm(self):
#		values = []
#		for i in range(len(self.arms)):
#			values.append(np.random.choice(self.arms[i]))
#		
#		arm_to_pull = np.argmax(values)

#		return arm_to_pull



class Bootrap_ts:
	def __init__(self, alpha, beta, size):
		self.alpha = alpha
		self.beta = beta
		self.size = size

		self.distribution = []
		self._alpha = []
		self._beta = []

		for i in range(self.size):
			self._alpha.append(self.alpha)
			self._beta.append(self.beta)

	def update(self, _return_):

		for i in range(self.size):
			if random.randint(0, 1) == 1:
				self._alpha[i] = self._alpha[i] + _return_
				self._beta[i] = self._beta[i] + (1 - _return_) 

	def getDist(self):
		self.distribution = []
		for i in range(self.size):
			alpha = self._alpha[i]
			beta = self._beta[i]
			val = alpha / (alpha + beta)
			self.distribution.append(val)

		return self.distribution	

class DMCTS_ts:
	def __init__(self, alpha, beta, size):
		self.alpha = alpha
		self.beta = beta
		self.size = size

		self.distribution = []
		self._alpha = []
		self._beta = []

		for i in range(self.size):
			self._alpha.append(self.alpha)
			self._beta.append(self.beta)

	def update(self, _return_):

		for i in range(self.size):
			if random.randint(0, 1) == 1:
				self._alpha[i] = self._alpha[i] + _return_
				self._beta[i] = self._beta[i] + (1) 

	def getDist(self):
		self.distribution = []
		for i in range(self.size):
			alpha = self._alpha[i]
			beta = self._beta[i]
			val = alpha / (beta)
			self.distribution.append(val)

		return self.distribution	


class Graph():
	def __init__(self, observations):
		self.obs = []
		self.observations = observations

		for i in range(self.observations):
			self.obs.append(i)

	def graph(self, _array_, _name_, _color_):

		i = 0
		stop = 1
		alpha_ = 0.2
		bins = []


		while i < 1:
			i += 0.001
			bins.append(i)

		

		
		if _name_ == "Emperical Distribution":
			plt.hist(_array_, range=[0, 1], facecolor = _color_, alpha = alpha_, label = _name_)
		else:
			plt.hist(_array_, range=[0, 1], facecolor = _color_, alpha = alpha_, label = _name_)
		#plt.plot(self.obs, sorted_arr, color = _color_, label = _name_)
		plt.xlabel('Utility')
		plt.ylabel('Frequency')
		plt.legend(loc='best')


def main():

	alpha = 1
	beta = 1
	#p_true = 0.5
	#p_false = 1 - p_true 
	arms = 5
	timestep = 1000
	size = 1000

	bandits = []
	_ptrue_ = [0.5, 0.6, 0.5, 0.9, 0.35]

	#bandit = Bandit(p_true, p_false)
	bts = Bootrap_ts(alpha, beta, size)
	dts = DMCTS_ts(alpha, beta, size)
	sampler = Sampler()
	grapher = Graph(size)

	for i in range(arms):
		print(i)
		bandits.append(Bandit(_ptrue_[i], 1 -_ptrue_[i]))

	returns = []
	average_utility = 0

	for i in range(timestep):
		return_ = bandit.pull_arm()
		

	for i in range(timestep):
		action = sampler.select_arm()
		return_ = bandits[action].pull_arm()
		sampler.update(return_, action)
		average_utility = (average_utility + return_) / timestep
		#print("utility", return_, "action", action, "average utility" ,average_utility)

	arm0 = sampler.getDist(0)
	print(arm0)
	arm1 = sampler.getDist(1)
	arm2 = sampler.getDist(2)
	arm3 = sampler.getDist(3)
	arm3 = sampler.getDist(4)

	#for i in range(timestep):
	#	_return_ = bandit.pull_arm()
	#	returns.append(_return_)
	#	bts.update(_return_)
		#sampler.update(_return_)
	#	dts.update(_return_)

	#print(sampler.getDist())
	#grapher.graph(bts.getDist(), "Bootstrap Thompson Sampling", "blue")
	#grapher.graph(sampler.getDist(), "Emperical Distribution", "red")
	#grapher.graph(dts.getDist(), "DMCTS Distribution", "magenta")
	grapher.graph(arm0, "Arm 0", "red")
	plt.show()
	grapher.graph(arm1, "Arm 1", "blue")
	plt.show()
	grapher.graph(arm2, "Arm 2", "green")
	plt.show()
	grapher.graph(arm3, "Arm 3", "magenta")
	plt.show()
	grapher.graph(arm3, "Arm 4", "yellow")
	plt.show()
	#plt.show()

if __name__ == "__main__":
    main()

