import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib
import itertools

class Plotter():

	def plot(self,table):
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

		fig = plt.figure()
		ax1 = fig.add_subplot(111, projection='3d')

		#ax1.bar3d(x, y, bottom, width, depth, top, shade=True)
		#ax1.bar3d(_x, _y, _z, 1, 1, dz, shade = True)
		ax1.plot_trisurf(_x, _y, dz, cmap='twilight_shifted')
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

		plt.show()
		#tikzplotlib.clean()
		#tikzplotlib.save("multivariate_cdf.pgf")


class Agent():
	def __init__(self):
		self.i = 1

	def stochastic_dominance(self, distribution1, distribution2):
		cond = False
		for vec in distribution1.vectors:
			a = distribution1.cdf_table[tuple(vec)]
			b = distribution2.cdf_table[tuple(vec)]
			if a < b:
				cond = True
			if a <= b:
				pass
			else:
				return False

		return cond


class Distribution:
	def __init__(self, max_val):
		self.max_val = max_val
		self.pdf_table = np.zeros([max_val + 1, max_val + 1])
		self.cdf_table = np.zeros([max_val + 1, max_val + 1])
		self.n = 0
		self.vectors = []

	def init_pdf(self, probs):

		for i in range(len(self.vectors)):
			self.pdf_table[tuple(np.array(self.vectors[i]))] = probs[i]
			 
		return self.pdf_table


	def update_pdf(self, vec):		
					
		pdf_table[tuple(np.array(vec))] += 1
		self.n += 1

		self.vectors.sort()
		self.vectors = list(self.vectors for self.vectors, _ in itertools.groupby(self.vectors))

		return self.pdf_table

	def update_cdf(self):
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

		return self.cdf_table


def main():

	max_val = 10
	plotter = Plotter()
	agent = Agent()
	distribution1 = Distribution(max_val)
	distribution2 = Distribution(max_val)

	distribution1.vectors = [[1, 1], [1, 5], [2, 3], [1, 2]]
	pdf_table1 = distribution1.init_pdf([0.1, 0.2, 0.5, 0.2])
	cdf_table1 = distribution1.update_cdf()

	distribution2.vectors = [[1, 1], [1, 2]]
	pdf_table2 = distribution2.init_pdf([0.5, 0.5])
	cdf_table2 = distribution2.update_cdf()

	dominance = agent.stochastic_dominance(distribution1, distribution2)
	print(dominance)

	dominance = agent.stochastic_dominance(distribution2, distribution1)
	print(dominance)
	
	plotter.plot(cdf_table1)
	plotter.plot(cdf_table2)


if __name__ == "__main__":
    main()

