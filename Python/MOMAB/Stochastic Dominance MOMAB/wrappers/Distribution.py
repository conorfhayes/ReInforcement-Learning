import numpy as np
import itertools

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
   		
		self.cdf_table = np.round(self.multidim_cdf(self.pdf_table/self.n), 2)
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