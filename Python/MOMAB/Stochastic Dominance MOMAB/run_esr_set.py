import numpy as np

import matplotlib as mpl

#mpl.use("pgf")
#pgf_with_pdflatex = {
#    "pgf.texsystem": "pdflatex",
#    "pgf.preamble": [
         #r"\usepackage[utf8x]{inputenc}",
         #r"\usepackage[T1]{fontenc}",
         #r"\usepackage{cmbright}",
#         ]
#}
#mpl.rcParams.update(pgf_with_pdflatex)

import matplotlib.pyplot as plt
import tikzplotlib
import itertools
import pandas as pd
import os
import sys
import time
import random


class Experiment():
	def __init__(self, bandits, agent, runs, exp_nums):
		self.bandits = bandits
		self.agent = agent
		self.runs = runs
		self.exp_nums = exp_nums

	def run(self):

		for run in range(self.runs):
			esr_vector = []
			esr_probs = []
			start = time.perf_counter()

		for i in range(self.exp_nums):
			if i == 0:
				for j in range(len(self.bandits)):
					_return_ = self.bandits[j].pull_arm()
					self.agent.update(j, _return_)

			action = self.agent.select_action()
			_return_ = self.bandits[action].pull_arm()
			self.agent.update(action, _return_)
			esr_index = self.agent.esr_dominance()

		for val in esr_index:
			esr_vector.append(self.agent.distribution[val].get_distribution()[0])
			esr_probs.append(self.agent.distribution[val].get_distribution()[1])


		end = time.perf_counter()
		print("")
		print('**** Run ' + str(run + 1) + ' - Execution Time: ' + str(round((end - start), 2)) +' seconds ****', )
		print(str(len(esr_index)) + " distributions in the ESR set")
		print("ESR Vector and Probabilities")
		print(esr_vector)
		print(esr_probs)
		#print(self.agent.distribution[0].update_cdf())
		#print(agent.distribution)
		print(" ")

		return


class Plotter():

	def plot(self, table,_type):
		_x = []
		_y = []
		probs = []

		# fake data
		for i in range(len(table)):
			for j in range(len(table)):
				_x.append(i)
				_y.append(j)
				probs.append(table[i][j])

		_z = np.zeros(len(probs))
		dx = np.ones(len(_x))
		dy = np.ones(len(_y))
		dz = probs

		data = {'x' : _x, 'y': _y, 'z': dz}
		df = pd.DataFrame(data)  

		fig = plt.figure()
		ax1 = fig.add_subplot(111, projection='3d')

		#ax1.bar3d(x, y, bottom, width, depth, top, shade=True)
		#ax1.bar3d(_x, _y, _z, 1, 1, dz, shade = True)
		ax1.plot_trisurf(_x, _y, dz, cmap='Blues')
		#ax1.plot_trisurf(_x, _y, dz, cmap='twilight_shifted')
		#ax1.scatter(_x, _y, dz, c=dz, cmap='BrBG', linewidth=1)

		ax1.set_xlabel('objective 1')
		ax1.set_ylabel('objective 2')
		ax1.set_zlabel('probability')

		ticks = [i for i in range(len(table))]

		ax1.set_xticks(ticks)
		ax1.set_yticks(ticks)
		ax1.set_zticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])

		ax1.axes.set_xlim3d(left=ticks[0], right=len(ticks) - 1) 
		ax1.axes.set_ylim3d(bottom=ticks[0], top=len(ticks) - 1) 
		ax1.axes.set_zlim3d(bottom=0, top=1) 

		#plt.show()

		filename = _type + '.csv'
		outdir = 'tikz/data/'
		print(os.path.join(os.getcwd(), outdir))
		path = os.path.normpath(os.path.join(os.getcwd(), outdir))
		if not os.path.exists(path):
			#print("Hello")
			os.makedirs(path, exist_ok = True)

		df.to_csv(path + '/' + _type + '.csv', index = False)
		#tikzplotlib.save("test.tex")
		#tikzplotlib.clean()
		#tikzplotlib.save("multivariate_cdf.pgf")
		#plt.savefig(_type)

		#filename = _type + '.csv'
		outdir = 'figures/'
		print(os.path.join(os.getcwd(), outdir))
		path = os.path.normpath(os.path.join(os.getcwd(), outdir))
		if not os.path.exists(path):
			#print("Hello")
			os.makedirs(path, exist_ok = True)

		plt.savefig(path + '/' + _type + ".png")	
		#tikzplotlib.save("multivariate_cdf.pgf")
		plt.show()


def main():
	from agents.Agent import Agent
	from envs.Bandit import Bandit
	from wrappers.Distribution import Distribution

	max_val = 10
	actions = 4
	episodes = 200000
	runs = 1
	plotter = Plotter()
	agent = Agent(actions, max_val)
	bandits = []

	#param = [True/False, number of obs in distribution, num objective, max value]
	bandits.append(Bandit(param = [True, 5, 2, 5]))
	bandits.append(Bandit(param = [True, 5, 2, 2]))
	bandits.append(Bandit(param = [True, 5, 2, 3]))
	#bandits.append(Bandit(param = [True, 5, 2, 5]))
	bandits.append(Bandit(param = [True, 5, 2, 2]))
	for i in range(actions):
		#bandits.append(Bandit(param = [True, 5, 2, 5]))
		print(bandits[i].vectors)
		print(bandits[i].probs)
		print(" ")

	experiment = Experiment(bandits, agent, runs, episodes)
	experiment.run()

	'''
	test code

	bandits.append(Bandit([[1, 1], [1, 5], [2, 3], [1, 2]], [0.1, 0.2, 0.5, 0.2]))
	bandits.append(Bandit([[1, 1], [1, 6], [2, 5], [1, 2]], [0.1, 0.2, 0.5, 0.2]))
	bandits.append(Bandit([[1, 1], [1, 5], [2, 3], [1, 2]], [0.1, 0.2, 0.5, 0.2]))
	bandits.append(Bandit([[1, 1], [1, 5], [2, 3], [1, 2]], [0.1, 0.2, 0.5, 0.2]))
	bandits.append(Bandit([[1, 1], [1, 5], [2, 3], [1, 2]], [0.1, 0.2, 0.5, 0.2]))
	bandits.append(Bandit([[1, 1], [1, 5], [2, 3], [1, 2]], [0.1, 0.2, 0.5, 0.2]))

	
	distribution_t1 = Distribution(max_val)
	distribution_t2 = Distribution(max_val)

	distribution_t1.vectors = [[1, 1], [1, 5], [2, 3], [1, 2]]
	pdf_table1 = distribution_t1.init_pdf([0.1, 0.2, 0.5, 0.2])
	cdf_table1 = distribution_t1.update_cdf()

	distribution_t2.vectors = [[1, 1], [1, 2]]
	pdf_table2 = distribution_t2.init_pdf([0.5, 0.5])
	cdf_table2 = distribution_t2.update_cdf()

	dominance = agent.stochastic_dominance(distribution_t1, distribution_t2)
	print(dominance)
	dominance = agent.stochastic_dominance(distribution_t2, distribution_t1)
	print(dominance)

	plotter.plot(pdf_table1, "pdf")	
	plotter.plot(cdf_table1, "cdf")
	plotter.plot(cdf_table2)
	'''


if __name__ == "__main__":
    main()

