
def main():
	from agents.Agent import Agent
	from envs.Bandit import Bandit
	from wrappers.Distribution import Distribution
	from experiments.Experiment import Experiment
	from plot.Plotter import Plotter
	from wrappers.SER import SER
	import argparse

	parser = argparse.ArgumentParser(description='')
	parser.add_argument('--max_val', default=10, type=int)
	#parser.add_argument('--num_bandits', default=1, type=int)
	parser.add_argument('--timesteps', default=100000, type=int)
	parser.add_argument('--runs', default=1, type=int)
	parser.add_argument('--type_bandits', default='random', type=str)
	args = parser.parse_args()
	print(args)

	max_val = args.max_val
	actions = 5
	episodes = args.timesteps
	runs = args.runs
	plotter = Plotter()
	agent = Agent(actions, max_val)
	
	bandits = []
	bandit_distributions = []
	bandit_probabilities = []


	if args.type_bandits == 'random':

		# Random Distributions
		#param = [True/False, number of obs in distribution, num objective, max value]
		bandits.append(Bandit(param = [True, 5, 2, 5]))
		bandits.append(Bandit(param = [True, 5, 2, 2]))
		bandits.append(Bandit(param = [True, 5, 2, 3]))
		bandits.append(Bandit(param = [True, 5, 2, 5]))
		bandits.append(Bandit(param = [True, 5, 2, 2]))
		print("*** Bandit Distributions ***")
		for i in range(actions):
			print(bandits[i].vectors)
			print(bandits[i].probs)
			print("")

	if args.type_bandits == 'bandit-example':

		# Manual Distributions
		bandits.append(Bandit([[2, 0], [2, 1]], [0.05, 0.05]))
		bandits.append(Bandit([[0, 0], [1, 1]], [0.1, 0.1]))
		bandits.append(Bandit([[1, 0], [1, 3]], [0.1, 0.1]))
		bandits.append(Bandit([[1, 0], [2, 1]], [0.1, 0.4]))
		bandits.append(Bandit([[1, 1], [1, 2]], [0.05, 0.05]))
		#bandits.append(Bandit([[1, 1], [1, 5], [2, 3], [1, 2]], [0.1, 0.2, 0.5, 0.2]))
		#bandits.append(Bandit([[1, 1], [1, 5], [2, 3], [1, 2]], [0.1, 0.2, 0.5, 0.2]))
		#bandits.append(Bandit([[1, 1], [1, 5], [2, 3], [1, 2]], [0.1, 0.2, 0.5, 0.2]))

	
	if args.type_bandits == 'manual-example':

		# Manual Distributions
		bandits.append(Bandit([[2, 0], [2, 1], [3, 2], [4, 2]], [0.05, 0.05, 0.1, 0.8]))
		bandits.append(Bandit([[0, 0], [1, 1], [2, 0], [2, 1]], [0.1, 0.1, 0.5, 0.3]))
		bandits.append(Bandit([[1, 0], [1, 3], [3, 4], [5, 4]], [0.1, 0.1, 0.2, 0.6]))
		bandits.append(Bandit([[1, 0], [2, 1], [3, 1], [3, 2]], [0.1, 0.4, 0.4, 0.1]))
		bandits.append(Bandit([[1, 1], [1, 2], [4, 0], [0, 0]], [0.05, 0.05, 0.1, 0.8]))
		#bandits.append(Bandit([[1, 1], [1, 5], [2, 3], [1, 2]], [0.1, 0.2, 0.5, 0.2]))
		#bandits.append(Bandit([[1, 1], [1, 5], [2, 3], [1, 2]], [0.1, 0.2, 0.5, 0.2]))
		#bandits.append(Bandit([[1, 1], [1, 5], [2, 3], [1, 2]], [0.1, 0.2, 0.5, 0.2]))


	for i in range(actions):
		#print(bandits[i].vectors)
		#print(bandits[i].probs)
		bandit_distributions.append(bandits[i].vectors)
		bandit_probabilities.append(bandits[i].probs)
		#print(" ")

	experiment = Experiment("bandit", bandits, agent, runs, episodes)
	experiment.run()
	#plotter.multi_cdf_plot(experiment.esr_vector, experiment.esr_probs)
	#plotter.multi_pdf_plot(experiment.esr_vector, experiment.esr_probs)
	#plotter.heatmap_plot([[2, 0], [2, 1], [3, 2], [4, 2]], [0.05, 0.05, 0.1, 0.8])
	#plotter.multi_heatmap_plot(experiment.esr_vector, experiment.esr_probs)
	#plotter.multi_joint_plot(experiment.esr_vector, experiment.esr_probs)
	#plotter.multi_pdf_bar_plot(experiment.esr_vector, experiment.esr_probs)


	'''
	test code

	
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