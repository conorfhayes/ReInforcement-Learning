**** Expected Scalarised Returns Dominance: A New Solution Concept for Multi-Objective Decision Making ****

This python code uses a Multi-Objective Distributional Reinforcement Learning algorithm to learn the ESR set when in the unknown utility function scenario for multi-objective multi-armed bandit problems.

The following python packages are required to run this code:
	matplotlib
	numpy
	pandas
	os
	sys
	time
	seaborn
	tikzplotlib
	itertools
	math
	
To run this code you need to specify the following arguments:
	--max_val
	--timesteps
	--runs
	--type_bandits

To run this code run the following command:
	python run_esr_set.py --max_val --timesteps --runs --type_bandits

For example:
 	python run_esr_set.py --max_val=10 --timesteps=100000 --runs=1 --type_bandits='realworld'


