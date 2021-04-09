
import os
import sys
import time
import numpy as np
#np.set_printoptions(threshold=sys.maxsize)
from wrappers.SER import SER
from plot.Plotter import Plotter
from metrics.Metrics import Metrics

class Experiment():
	def __init__(self, _type, bandits, agent, runs, exp_nums, k_vec, k_probs):
		self._type = _type
		self.esr_vector = []
		self.esr_probs = []
		self.f1_score = []
		self.plotter = Plotter()
		self.metrics = Metrics()
		self.k_vec = k_vec
		self.k_probs = k_probs

		if self._type == "bandit":
			self.bandits = bandits
			self.agent = agent
			self.runs = runs
			self.exp_nums = exp_nums

	def run(self):
		if self._type == "bandit":
			for run in range(self.runs):
				start = time.perf_counter()

				for i in range(self.exp_nums):
					self.esr_vector = []
					self.esr_probs = []
					if i == 0:
						for j in range(len(self.bandits)):
							_return_ = self.bandits[j].pull_arm()
							self.agent.update(j, _return_)

					action = self.agent.select_action()
					_return_ = self.bandits[action].pull_arm()
					self.agent.update(action, _return_)
					esr_index = self.agent.esr_dominance()

					for val in esr_index:
						self.esr_vector.append(self.agent.distribution[val].get_distribution()[0])
						self.esr_probs.append(self.agent.distribution[val].get_distribution()[1])

					self.f1_score.append(self.metrics.precision_recall(self.esr_vector, self.esr_probs, self.k_vec, self.k_probs))

				end = time.perf_counter()
				self.f1_score = np.mean(np.array(self.f1_score).reshape(-1, 1000), axis=1)

				ser = SER(self.esr_vector, self.esr_probs)
				ser_expectations = ser.expectations()
				ser_pareto_front = ser.pareto_front()
				print("")
				print('**** Run ' + str(run + 1) + ' - Execution Time: ' + str(round((end - start), 2)) +' seconds ****', )
				print(str(len(esr_index)) + " distributions in the ESR set")
				print("ESR Vector and Probabilities")
				for a in range(len(self.esr_vector)):
					print(self.esr_vector[a])
					print(self.esr_probs[a])
					print(" ")
				#print(self.f1_score)
				print("")
				print("SER - Pareto Front")
				print("Number of policies on the pareto front : " + str(len(ser_pareto_front)))
				print(ser_pareto_front)
				print("")

				self.plotter.multi_heatmap_plot(self.esr_vector, self.esr_probs)
				self.plotter.multi_pdf_bar_plot(self.esr_vector, self.esr_probs)				
				self.plotter.multi_cdf_plot(self.esr_vector, self.esr_probs)
				self.plotter.xy_plot(self.f1_score, self.exp_nums)

				#print(self.agent.distribution[0].update_cdf())
				#print(agent.distribution)
				

		return 