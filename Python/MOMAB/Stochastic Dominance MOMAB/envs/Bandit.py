import numpy as np
import random

class Bandit():
	# param = [True/False, number of obs in distribution, num objective, max value]
	def __init__(self, vectors=False, probs=False, param=False):
		self.n = 1
		self.param = param

		if self.param == False:
			self.vectors = vectors
			self.probs = probs
		else:
			self.generateRandomBandit()
			

	def pull_arm(self):
			
		return self.vectors[np.random.choice(len(self.vectors), 1, replace=False, p=self.probs)[0]]

	def generateRandomBandit(self):
		objectives = self.param[2]
		mass = 1
		obs = []
		probs = []
		number = random.randint(2, self.param[1])
		for i in range(self.param[1]):
			vec = []
			for j in range(objectives):
				vec.append(random.randint(0,self.param[3]))
			probs.append(random.randint(1,self.param[3]))
			obs.append(vec)

		sum_probs = np.sum(probs)
		probs = probs/sum_probs
		self.vectors = obs
		self.probs = probs
		#print(probs)
		#print(sum(probs))
