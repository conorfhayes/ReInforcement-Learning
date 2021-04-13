import matplotlib.pyplot as plt
import matplotlib as mpl
import tikzplotlib
import itertools
import pandas as pd
import numpy as np
import seaborn as sns
import os
import sys
import time
import random

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

class Plotter():

	def __init__(self, vec, probs, logdir, exp, single = False, save = False):
		self.vec = vec
		self.probs = probs
		self.single = single
		self.save = save
		self.logdir = logdir
		self.exp = exp

	#self.colors = [
    #        'Reds', 'Greens', 'Blues', 'Purples', 'Oranges', 'Greys',
    #        'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
    #        'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']


	def multidim_cdf(self, pdf):
		cdf = pdf[...,::1].cumsum(1)[...,::1]
		for i in range(2,pdf.ndim+1):
			np.cumsum(cdf, axis=-i, out=cdf)
		return cdf

	def heatmap_plot(self, table_vec, table_probs, i):
		self.colors = [
            'Reds', 'Greens', 'Blues', 'Purples', 'Oranges', 'Greys',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

		max_val = 10
		fig = plt.figure()
		ax1 = fig.add_subplot(111)
		pdf_table = np.zeros([max_val + 1, max_val + 1])		

		for j in range(len(table_vec)):
			pdf_table[tuple(np.array(table_vec[j]))] = table_probs[j]			

		im = ax1.imshow(pdf_table, cmap= self.colors[i])

		ax1.set_xlabel('objective 1')
		ax1.set_ylabel('objective 2')

		ticks = [i for i in range(max_val + 1)]

		ax1.set_xticks(ticks)
		ax1.set_yticks(ticks)
		#ax1.set_zticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])

		ax1.axes.set_xlim(left=ticks[0], right=len(ticks) - 1) 
		ax1.axes.set_ylim(bottom=ticks[0], top=len(ticks) - 1) 
		#ax1.colorbar()
		#ax1.axes.set_zlim3d(bottom=0, top=1) 
		cbar = fig.colorbar(im, orientation='vertical')
		#cbar.set_ticks(np.arange(0, 1, 0.1))


		if self.save == True:
			plt.savefig(self.logdir + '/' + "heatmap_distribution_" + str(i) + ".png")	
			#tikzplotlib.save(self.logdir + '/' + "heatmap_distribution_" + str(i) + ".pgf")

			mpl.use("pgf")
			pgf_with_pdflatex = {
			"pgf.texsystem": "pdflatex",
			"pgf.preamble": [
	        r"\usepackage[utf8x]{inputenc}",
	        r"\usepackage[T1]{fontenc}",
	        r"\usepackage{cmbright}",
			]
			}
			mpl.rcParams.update(pgf_with_pdflatex)
			plt.savefig(self.logdir + '/' + "heatmap_distribution_" + str(i) + ".pdf")	

		else:
			plt.show()
		


		

		return

	def multi_heatmap_plot(self, table_vec, table_probs):
		self.colors = [
            'Reds', 'Greens', 'Blues', 'Purples', 'Oranges', 'Greys',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

		max_val = 10
		fig = plt.figure()
		ax1 = fig.add_subplot(111)
		pdf_table = np.zeros([max_val + 1, max_val + 1])
		
		for i in range(len(table_vec)):
			#_x = []
			#_y = []
			#cdf_p = []
			
			vec = table_vec[i]
			probs = table_probs[i]

			for j in range(len(table_vec[i])):
				pdf_table[tuple(np.array(vec[j]))] = probs[j]			
		

		im = ax1.imshow(pdf_table, cmap='Blues')

		ax1.set_xlabel('objective 1')
		ax1.set_ylabel('objective 2')
		#ax1.set_zlabel('probability')

		ticks = [i for i in range(max_val + 1)]

		ax1.set_xticks(ticks)
		ax1.set_yticks(ticks)
		#ax1.set_zticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])

		ax1.axes.set_xlim(left=ticks[0], right=len(ticks) - 1) 
		ax1.axes.set_ylim(bottom=ticks[0], top=len(ticks) - 1) 
		#ax1.colorbar()
		#ax1.axes.set_zlim3d(bottom=0, top=1) 
		cbar = fig.colorbar(im, orientation='vertical')
		#cbar.set_ticks(np.arange(0, 1, 0.1))

		if self.save == True:
			plt.savefig(self.logdir + '/' + "multi_heatmap" + ".png")

			mpl.use("pgf")
			pgf_with_pdflatex = {
			"pgf.texsystem": "pdflatex",
			"pgf.preamble": [
	        r"\usepackage[utf8x]{inputenc}",
	        r"\usepackage[T1]{fontenc}",
	        r"\usepackage{cmbright}",
			]
			}
			mpl.rcParams.update(pgf_with_pdflatex)	
			plt.savefig(self.logdir + '/' + "multi_heatmap" + ".pdf")	
			#tikzplotlib.save("multivariate_cdf.pgf")

		else:
			plt.show()


		return

	def cdf_plot(self, table_vec, table_probs, i):
		self.colors = [
            'Reds', 'Greens', 'Blues', 'Purples', 'Oranges', 'Greys',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

		max_val = 10
		fig = plt.figure()
		ax1 = fig.add_subplot(111, projection='3d')

		
		_x = []
		_y = []
		cdf_p = []
		pdf_table = np.zeros([max_val + 1, max_val + 1])
		cdf_table = np.zeros([max_val + 1, max_val + 1])

		for j in range(len(table_vec)):
			vec = table_vec[j]
			probs = table_probs[j]
			pdf_table[tuple(np.array(vec))] = probs

		cdf_table = self.multidim_cdf(pdf_table)

		
		for a in range(len(cdf_table)):
			for b in range(len(cdf_table)):
				_x.append(a)
				_y.append(b)
				cdf_p.append(cdf_table[a][b])

		_z = np.zeros(len(cdf_p))
		dx = np.ones(len(_x))
		dy = np.ones(len(_y))
		dz = cdf_p
	

		ax1.plot_trisurf(_x, _y, dz, cmap = self.colors[i], alpha = 0.6)

		ax1.set_xlabel('objective 1')
		ax1.set_ylabel('objective 2')
		ax1.set_zlabel('probability')

		ticks = [i for i in range(max_val + 1)]

		ax1.set_xticks(ticks)
		ax1.set_yticks(ticks)
		ax1.set_zticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])

		ax1.axes.set_xlim3d(left=ticks[0], right=len(ticks) - 1) 
		ax1.axes.set_ylim3d(bottom=ticks[0], top=len(ticks) - 1) 
		ax1.axes.set_zlim3d(bottom=0, top=1) 

		if self.save == True:
			plt.savefig(self.logdir + '/' + "cdf_distribution_" + str(i) + ".png")

			mpl.use("pgf")
			pgf_with_pdflatex = {
			"pgf.texsystem": "pdflatex",
			"pgf.preamble": [
	        r"\usepackage[utf8x]{inputenc}",
	        r"\usepackage[T1]{fontenc}",
	        r"\usepackage{cmbright}",
			]
			}
			mpl.rcParams.update(pgf_with_pdflatex)	
			plt.savefig(self.logdir + '/' + "cdf_distribution_" + str(i) + ".pdf")	
			#tikzplotlib.save("multivariate_cdf.pgf")

		else:
			plt.show()
		



		return


	def multi_cdf_plot(self, table_vec, table_probs):
		self.colors = [
            'Reds', 'Greens', 'Blues', 'Purples', 'Oranges', 'Greys',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

		max_val = 10
		fig = plt.figure()
		ax1 = fig.add_subplot(111, projection='3d')

		for i in range(len(table_vec)):
			_x = []
			_y = []
			cdf_p = []
			pdf_table = np.zeros([max_val + 1, max_val + 1])
			cdf_table = np.zeros([max_val + 1, max_val + 1])

			for j in range(len(table_vec[i])):
				vec = table_vec[i]
				probs = table_probs[i]
				pdf_table[tuple(np.array(vec[j]))] = probs[j]
			cdf_table = self.multidim_cdf(pdf_table)

			
			for a in range(len(cdf_table)):
				for b in range(len(cdf_table)):
					_x.append(a)
					_y.append(b)
					cdf_p.append(cdf_table[a][b])

			_z = np.zeros(len(cdf_p))
			dx = np.ones(len(_x))
			dy = np.ones(len(_y))
			dz = cdf_p
		

			ax1.plot_trisurf(_x, _y, dz, cmap = self.colors[i], alpha = 0.6)

		ax1.set_xlabel('objective 1')
		ax1.set_ylabel('objective 2')
		ax1.set_zlabel('probability')

		ticks = [i for i in range(max_val + 1)]

		ax1.set_xticks(ticks)
		ax1.set_yticks(ticks)
		ax1.set_zticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])

		ax1.axes.set_xlim3d(left=ticks[0], right=len(ticks) - 1) 
		ax1.axes.set_ylim3d(bottom=ticks[0], top=len(ticks) - 1) 
		ax1.axes.set_zlim3d(bottom=0, top=1) 


		if self.save == True:
			plt.savefig(self.logdir + '/' + "multi_cdf_distribution" + ".png")

			mpl.use("pgf")
			pgf_with_pdflatex = {
			"pgf.texsystem": "pdflatex",
			"pgf.preamble": [
	        r"\usepackage[utf8x]{inputenc}",
	        r"\usepackage[T1]{fontenc}",
	        r"\usepackage{cmbright}",
			]
			}
			mpl.rcParams.update(pgf_with_pdflatex)	
			plt.savefig(self.logdir + '/' + "multi_cdf_distribution" + ".pdf")	
			#tikzplotlib.save("multivariate_cdf.pgf")

		else:
			plt.show()


		return

	def multi_pdf_surface_plot(self, table_vec, table_probs):
		self.colors = [
            'Reds', 'Greens', 'Blues', 'Purples', 'Oranges', 'Greys',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
		max_val = 10
		fig = plt.figure()
		ax1 = fig.add_subplot(111, projection='3d')

		for i in range(len(table_vec)):
			_x = []
			_y = []
			pdf_p = []
			pdf_table = np.zeros([max_val + 1, max_val + 1])
			

			for j in range(len(table_vec[i])):
				vec = table_vec[i]
				probs = table_probs[i]
				pdf_table[tuple(np.array(vec[j]))] = probs[j]

			
			for a in range(len(pdf_table)):
				for b in range(len(pdf_table)):
					_x.append(a)
					_y.append(b)
					pdf_p.append(pdf_table[a][b])

			_z = np.zeros(len(pdf_p))
			dx = np.ones(len(_x))
			dy = np.ones(len(_y))
			dz = pdf_p
		

			ax1.plot_trisurf(_x, _y, dz, cmap = self.colors[i], alpha = 0.6)

		ax1.set_xlabel('objective 1')
		ax1.set_ylabel('objective 2')
		ax1.set_zlabel('probability')

		ticks = [i for i in range(max_val + 1)]

		ax1.set_xticks(ticks)
		ax1.set_yticks(ticks)
		ax1.set_zticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])

		ax1.axes.set_xlim3d(left=ticks[0], right=len(ticks) - 1) 
		ax1.axes.set_ylim3d(bottom=ticks[0], top=len(ticks) - 1) 
		ax1.axes.set_zlim3d(bottom=0, top=1) 

		plt.show()


		return

	def multi_pdf_bar_plot(self, table_vec, table_probs):
		self.colors = ['r', 'g', 'b']
		max_val = 10
		fig = plt.figure()
		ax1 = fig.add_subplot(111, projection='3d')

		for i in range(len(table_vec)):
			_x = []
			_y = []
			pdf_p = []
			pdf_table = np.zeros([max_val + 1, max_val + 1])
			

			for j in range(len(table_vec[i])):
				vec = table_vec[i]
				probs = table_probs[i]
				pdf_table[tuple(np.array(vec[j]))] = probs[j]

			
			for a in range(len(pdf_table)):
				for b in range(len(pdf_table)):
					_x.append(a)
					_y.append(b)
					pdf_p.append(pdf_table[a][b])

			_z = np.zeros(len(pdf_p))
			dx = np.ones(len(_x))
			dy = np.ones(len(_y))
			dz = pdf_p
			df = pd.DataFrame({'_x': _x, '_y': _y, '_z':_z, 'dx':dx, 'dy':dy, 'dz':dz})
			df = df[df['dz'] !=0]
			ax1.bar3d(df['_x'], df['_y'], df['_z'], df['dx'], df['dy'], df['dz'], alpha = 0.6)

		ax1.set_xlabel('objective 1')
		ax1.set_ylabel('objective 2')
		ax1.set_zlabel('probability')

		ticks = [i for i in range(max_val + 1)]

		ax1.set_xticks(ticks)
		ax1.set_yticks(ticks)
		ax1.set_zticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])

		ax1.axes.set_xlim3d(left=ticks[0], right=len(ticks) - 1) 
		ax1.axes.set_ylim3d(bottom=ticks[0], top=len(ticks) - 1) 
		ax1.axes.set_zlim3d(bottom=0, top=1) 

		plt.show()


		return

	def multi_3d_pdf_bar_plot(self, table_vec, table_probs):
		self.colors = ['r', 'g', 'b']
		max_val = 10
		#fig = plt.figure()
		ax1 = plt.figure().add_subplot(projection='3d')

		for i in range(len(table_vec)):
			#if i >= 1:
			#	pass
			#else:
			_x = []
			_y = []
			pdf_p = []
			pdf_table = np.zeros([max_val + 1, max_val + 1])
			

			for j in range(len(table_vec[i])):
				vec = table_vec[i]
				probs = table_probs[i]
				pdf_table[tuple(np.array(vec[j]))] = probs[j]

			
			for a in range(len(pdf_table)):
				for b in range(len(pdf_table)):
					_x.append(a)
					_y.append(b)
					pdf_p.append(pdf_table[a][b])

			_z = np.zeros(len(pdf_p))
			dx = np.ones(len(_x))
			dy = np.ones(len(_y))
			dz = pdf_p
			df = pd.DataFrame({'_x': _x, '_y': _y, '_z':_z, 'dx':dx, 'dy':dy, 'dz':dz})
			df = df[df['dz'] !=0]
			#print(df)
			#ax1.bar3d(df['_x'], df['_y'], df['_z'], df['dx'], df['dy'], df['dz'], alpha = 0.6)
			ax1.bar(df['_x'], df['dz'], zs=0, zdir='x', color = self.colors[i], alpha = 0.4)
			ax1.bar(df['_y'], df['dz'], zs=0, zdir='y', color = self.colors[i], alpha = 0.4)
			ax1.scatter(df['_x'], df['_y'], zs=0, zdir='z', color = self.colors[i], alpha = 0.4)

		ax1.set_xlabel('objective 1')
		ax1.set_ylabel('objective 2')
		ax1.set_zlabel('probability')

		ticks = [i for i in range(max_val + 1)]

		ax1.set_xticks(ticks)
		ax1.set_yticks(ticks)
		ax1.set_zticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])

		ax1.axes.set_xlim3d(left=ticks[0], right=len(ticks) - 1) 
		ax1.axes.set_ylim3d(bottom=ticks[0], top=len(ticks) - 1) 
		ax1.axes.set_zlim3d(bottom=0, top=1) 

		plt.show()


		return

	def man_plot(self, vec, probs, _type):
		self.vec = vec
		self.probs = probs
		_x = []
		_y = []
		self.cdf_p = []

		if _type == "pdf":

			for i in range(len(self.vec)):			
				_x.append(i[0])
				_y.append(j[1])

		if _type == "cdf":
			max_val = 10
			pdf_table = np.zeros([max_val + 1, max_val + 1])
			cdf_table = np.zeros([max_val + 1, max_val + 1])
			for i in range(len(self.vec)):
				pdf_table[tuple(np.array(self.vec[i]))] = self.probs[i]
			cdf_table = self.multidim_cdf(pdf_table)

			for i in range(len(cdf_table)):
				for j in range(len(cdf_table)):
					_x.append(i)
					_y.append(j)
					self.cdf_p.append(cdf_table[i][j])

			print(cdf_table)
				

		_z = np.zeros(len(self.cdf_p))
		dx = np.ones(len(_x))
		dy = np.ones(len(_y))
		dz = self.cdf_p

		fig = plt.figure()
		ax1 = fig.add_subplot(111, projection='3d')

		ax1.plot_trisurf(_x, _y, dz, cmap='Blues')

		ax1.set_xlabel('objective 1')
		ax1.set_ylabel('objective 2')
		ax1.set_zlabel('probability')

		ticks = [i for i in range(max_val + 1)]

		ax1.set_xticks(ticks)
		ax1.set_yticks(ticks)
		ax1.set_zticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])

		ax1.axes.set_xlim3d(left=ticks[0], right=len(ticks) - 1) 
		ax1.axes.set_ylim3d(bottom=ticks[0], top=len(ticks) - 1) 
		ax1.axes.set_zlim3d(bottom=0, top=1) 

		plt.show()

	def f1_plot(self, dir_):
		inc = 100
		dataframe = pd.read_csv(dir_ + 'f1_score.csv')
		_y = dataframe['mean']
		_y = np.mean(np.array(_y).reshape(-1, inc), axis=1)

		#_y = data
		#exp = len(_y)
		#_x = np.arange(0, exp + inc, inc)
		fig = plt.figure()
		ax1 = fig.add_subplot(111)

		ax1.plot(dataframe['mean'], color='lightblue')
		ax1.set_xlabel('Episodes')
		ax1.set_ylabel('F1 Score')

		ticks = np.arange(0, len(dataframe['mean']) + len(dataframe['mean'])/10 , len(dataframe['mean'])/10)
		ax1.set_xticks(ticks)
		#ax1.set_yticks(ticks)

		if self.save == True:
			plt.savefig(self.logdir + '/' + "f1_plot" + ".png")

			mpl.use("pgf")
			pgf_with_pdflatex = {
			"pgf.texsystem": "pdflatex",
			"pgf.preamble": [
	        r"\usepackage[utf8x]{inputenc}",
	        r"\usepackage[T1]{fontenc}",
	        r"\usepackage{cmbright}",
			]
			}
			mpl.rcParams.update(pgf_with_pdflatex)	
			plt.savefig(self.logdir + '/' + "f1_plot" + ".pdf")	
			#tikzplotlib.save("multivariate_cdf.pgf")

		else:
			plt.show()
		return

	

	def plot_run(self):
		
		#cdfs
		
		a = 0
		for i in range(len(self.vec)):
			self.cdf_plot(self.vec[i], self.probs[i], a)
			a += 1
		self.multi_cdf_plot(self.vec, self.probs)

		#heatmaps
		a = 0
		for i in range(len(self.vec)):
			self.heatmap_plot(self.vec[i], self.probs[i], a)
			a += 1
		self.multi_heatmap_plot(self.vec, self.probs)

		#f1 score
		self.f1_plot(self.logdir)



