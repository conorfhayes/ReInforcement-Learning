import numpy as np
import itertools
import math

class SER:
	def __init__(self, table_vec, table_probs):
		self.table_vec = table_vec
		self.table_probs = table_probs

	def expectations(self):

		self.expecations = []
		for i in range(len(self.table_vec)):
			val = np.array([0.0,0.0])
			for j in range(len(self.table_vec[i])):
				val += np.array(self.table_vec[i][j]) * np.round(self.table_probs[i][j], 2)
			self.expecations.append(val)		

			 
		return self.expecations

	def pareto_front(self):
		p = 0
		
		self.pareto_set = []
		for i in self.expecations:
			a = 0
			inSet = True
			arm = i
			b = 0
			for j in self.expecations:
				if a == b:
					dominated = False
				else:					
					dominated = self.check_pareto_domination(j, i)

				if dominated == True:
					inSet = False
					break
				b += 1

			if inSet == True:
				self.pareto_set.append(i)
			a += 1

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


	