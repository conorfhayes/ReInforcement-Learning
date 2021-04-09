
import os
import sys
import time
import numpy as np

class Metrics():
	def __init__(self):
		self.e = 0.01

	def regret(self):
		

		return 
	

	def precision_recall_compare(self, f_vec, f_probs, esr_vec, esr_probs):
		a =[] 
		b = []
		for i in range(len(esr_vec)):
			if len(f_vec) == len(esr_vec[i]):
				vec = np.isclose(f_vec, esr_vec[i], atol = self.e)
				probs = np.isclose(f_probs, esr_probs[i], atol = self.e)

				if vec.all() == True and probs.all() == True:
					return f_vec, f_probs

		return a, b

	def precision_recall(self, f_vec, f_probs, esr_vec, esr_probs):
		self.f_vec = f_vec
		self.f_probs = f_probs
		self.esr_vec = esr_vec
		self.esr_probs = esr_probs
		self.F_vec = []
		self.F_probs = []
		self.holder_vecs = []
		self.holder_probs = []
		#print(self.esr_vec)

		for i in range(len(self.f_vec)):
			comp = self.precision_recall_compare(self.f_vec[i], self.f_probs[i], self.esr_vec, self.esr_probs)
			vec = comp[0]
			probs = comp[0]

			if len(vec) == 0:
				pass
			else:
				self.F_vec.append(vec)
				self.F_probs.append(probs)

		precision = (len(self.F_vec) / len(self.f_vec))
		recall = (len(self.F_vec) / len(self.esr_vec))
		if precision + recall == 0:
			f1 = 0
		else:
			f1 = 2 * ((precision * recall) / (precision + recall))


		return f1