
import os
import sys
import time
import numpy as np
import pandas as pd
#np.set_printoptions(threshold=sys.maxsize)
from wrappers.SER import SER
from plot.Plotter import Plotter
from metrics.Metrics import Metrics
from agents.Agent import Agent
from copy import deepcopy

class Experiment():
	def __init__(self, _type, bandits, agent, runs, exp_nums, logdir, k_vec, k_probs):
		self._type = _type
		self.esr_vector = []
		self.esr_probs = []
		self.f1_score = []
		self.f1 = []
		#self.plotter = Plotter()
		self.metrics = Metrics()
		self.k_vec = k_vec
		self.k_probs = k_probs
		self.f1_df = pd.DataFrame()
		self.esrBandit = Agent(0, 10)
		self.logdir = logdir

		for i in range(len(k_vec)):
			self.esrBandit.manual_distribution(k_vec[i], k_probs[i])


		if self._type == "bandit":
			self.bandits = bandits
			self.agent = agent
			self.runs = runs
			self.exp_nums = exp_nums

	def run(self):
		if self._type == "bandit":
			avg_log = self.logdir + 'average' + '/'
			if not os.path.exists(avg_log):
					os.makedirs(avg_log, exist_ok = True)

			for run in range(self.runs):
				self.run_df = pd.DataFrame()
				start = time.perf_counter()
				run_log = self.logdir + 'run_' + str(run + 1) + '/'

				if not os.path.exists(run_log):
					os.makedirs(run_log, exist_ok = True)

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
					
					self.esr_agent = deepcopy(self.agent)
					self.esr_agent.distribution = np.array(self.esr_agent.distribution)[esr_index]

					for val in esr_index:
						self.esr_vector.append(self.agent.distribution[val].get_distribution()[0])
						self.esr_probs.append(self.agent.distribution[val].get_distribution()[1])

					#self.f1.append(self.metrics.precision_recall(self.esr_vector, self.esr_probs, self.k_vec, self.k_probs))
					self.f1.append(self.metrics.pr_kl(self.esr_agent, self.esrBandit))


				self.run_df['run' + str(run)] = self.f1
				self.run_df['mean'] = self.run_df.mean(axis=1)
				self.f1_df['run' + str(run)] = self.f1
				end = time.perf_counter()
				
				#self.run_df['average'] = self.f1_df.mean(axis=1)
				#print(self.f1_df)
				#self.f1_score = self.f1_df['Average']
				#self.run_df['average'] = np.mean(np.array(self.f1_score).reshape(-1, 10), axis=1)
				self.run_df.to_csv(run_log + "/f1_score.csv", index = False)

				ser = SER(self.k_vec, self.k_probs)
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
				print("")
				print("SER - Pareto Front")
				print("Number of policies on the pareto front : " + str(len(ser_pareto_front)))
				print(ser_pareto_front)
				print("")

				self.plotter = Plotter(self.esr_vector, self.esr_probs, run_log, self.exp_nums, True, True)
				self.plotter.plot_run()

			self.f1_df['mean'] = self.f1_df.mean(axis=1)
			#self.f1_df['average'] = np.mean(np.array(self.f1_df['mean']).reshape(-1, 10), axis=1)
			self.f1_df.to_csv(avg_log + "/f1_score.csv", index = False)
			self.plotter = Plotter(self.esr_vector, self.esr_probs, avg_log, self.exp_nums, True, True)
			self.plotter.plot_run()


	
				

		return 