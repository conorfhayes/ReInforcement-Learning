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
			for j in range(len(self.arms)):
				if j == i:
					dominated = False
				else:
					dominated = self.stochastic_dominance(self.distribution[j], self.distribution[i])
				if dominated == True:
					inSet = False
					break
			if inSet == True:
				self.esr_set.append(i)

		return self.esr_set



class Distribution:
	def __init__(self, max_val):
		self.max_val = max_val
		self.pdf_table = np.zeros([max_val + 1, max_val + 1])
		self.cdf_table = np.zeros([max_val + 1, max_val + 1])
		self.n = 1
		self.vectors = []

	def init_pdf(self, probs):

		for i in range(len(self.vectors)):
			self.pdf_table[tuple(np.array(self.vectors[i]))] = probs[i]
			 
		return self.pdf_table


	def update_pdf(self, vec):		
					
		self.pdf_table[tuple(np.array(vec))] += 1
		self.n += 1

		self.vectors.append(vec)
		self.vectors.sort()
		self.vectors = list(self.vectors for self.vectors, _ in itertools.groupby(self.vectors))

		return self.pdf_table

	def get_distribution(self):

		probs = []
		for vec in self.vectors:
			probs.append(self.pdf_table[tuple(np.array(vec))]/self.n)

		return self.vectors, probs

	def multidim_cdf(self, pdf):
			cdf = pdf[...,::1].cumsum(1)[...,::1]
			for i in range(2,pdf.ndim+1):
				np.cumsum(cdf, axis=-i, out=cdf)
			return cdf

	def update_cdf(self):		
   		
		self.cdf_table = np.round(self.multidim_cdf(self.pdf_table/self.n), 1)
		'''
		for i in range(len(self.cdf_table)):
			for j in range(len(self.cdf_table)):
				cdf = 0
				for x in range(len(self.pdf_table)):
					for y in range(len(self.pdf_table)):
						if x <= i and y <= j:
							cdf += self.pdf_table[x][y]
						else:
							break
				self.cdf_table[i][j] = cdf
		'''
		#print(self.cdf_table)

		return self.cdf_table



def main():

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

