from wrappers.Distribution import Distribution
import numpy as np

class Agent():
	def __init__(self, actions, max_val):
		self.i = 1
		self.max_val = max_val
		self.actions = actions
		self.esr_set = []
		self.distribution = []
		self.arms = []

		for action in range(self.actions):
			self.distribution.append(Distribution(self.max_val))
			self.arms.append(action)


	def stochastic_dominance(self, distribution1, distribution2):

		cond1 = (distribution1.cdf_table <= distribution2.cdf_table).all()
		cond2 = (distribution1.cdf_table < distribution2.cdf_table).any()	

		if cond1 == True and cond2 == True:
			return True
		else:
			return False

	def select_action(self):
		return np.random.choice(self.arms)

	def update(self, action, _return_):
		self.distribution[action].update_pdf(_return_)
		self.distribution[action].update_cdf()

	def esr_dominance(self):
		self.esr_set = []

		for i in range(len(self.arms)):
			inSet = True
			dominated = False
			arm = self.distribution[i]
			for j in range(len(self.arms)):
				if i == j:
					pass
				else:
					dominated = self.stochastic_dominance(self.distribution[j], arm)
				if dominated == True:
					inSet = False
					break
			if inSet == True:
				self.esr_set.append(i)

		return self.esr_set

	def manual_distribution(self, vec, probs):
		
		self.distribution.append(Distribution(self.max_val))	
		for i in range(len(vec)):				
			self.distribution[-1].manual_pdf(vec[i], probs[i])
		
		self.distribution[-1].update_cdf()

		return