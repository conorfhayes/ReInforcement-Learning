
import os
import sys
import time
import numpy as np
import math as math

class Metrics():
	def __init__(self):
		self.e = 0.01

	def regret(self):
		

		return 

	def kl_divergence(self, pdf1, pdf2):
		#return np.abs(np.sum(np.where((pdf1 != 0) & (pdf2 != 0), pdf1 * np.log(pdf1 / pdf2), 0)))
		val = 0
		for i in range(len(pdf1)):
			for j in range(len(pdf1[i])):
				if pdf1[i][j] == 0 and pdf2[i][j] == 0:
					val += 0
				elif pdf1[i][j] == 0:
					val += 0.1
				elif pdf2[i][j] == 0:
					val += 0.1
				else:
					
					val += pdf1[i][j] * np.log(pdf1[i][j] / pdf2[i][j])

		return np.abs(val)

	def ks_statistic(self, cdf1, cdf2):

		val = np.max(cdf1 - cdf2)

		return val

	def precision_recall_compare(self, f_vec, f_probs, esr_vec, esr_probs):
		a = [] 
		b = []
		for i in range(len(esr_vec)):
			if len(f_vec) == len(esr_vec[i]):
				vec = np.isclose(f_vec, esr_vec[i], atol = self.e)
				probs = np.isclose(f_probs, esr_probs[i], atol = self.e)

				if vec.all() == True and probs.all() == True:
					return f_vec, f_probs

		return a, b

	def precision_recall_kl(self, agent, man_agent):

		for i in range(len(man_agent.distribution)):
			kl_score = self.kl_divergence(agent.pdf_table/agent.n, man_agent.distribution[i].pdf_table)
			if kl_score <= self.e:
				return 1

		return 0

	def precision_recall_ks(self, agent, man_agent):
		for i in range(len(man_agent.distribution)):
			ks_score = self.ks_statistic(np.round(agent.cdf_table, 2), np.round(man_agent.distribution[i].cdf_table, 2))
			if ks_score <= self.e:
				return 1

		return 0

	def pr_kl(self, agent, man_agent):
		self.agent = agent
		self.man_agent = man_agent
		self.f = len(self.agent.distribution)
		self.esr = len(self.man_agent.distribution)
		self.F = 0

		for i in range(len(self.agent.distribution)):
			kl_div = self.precision_recall_ks(self.agent.distribution[i], self.man_agent)			

			if kl_div == 1:
				self.F += 1

		if self.f == 0:
			precision = 0
		else:
			precision = self.F / self.f
		recall = self.F / self.esr

		if precision + recall == 0:
			f1 = 0
		else:
			f1 = 2 * ((precision * recall) / (precision + recall))


		return f1

	def precision_recall(self, f_vec, f_probs, esr_vec, esr_probs):
		self.f_vec = f_vec
		self.f_probs = f_probs
		self.esr_vec = esr_vec
		self.esr_probs = esr_probs
		self.F_vec = []
		self.F_probs = []
		self.holder_vecs = []
		self.holder_probs = []

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