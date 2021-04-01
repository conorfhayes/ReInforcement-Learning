import matplotlib.pyplot as plt
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

	#self.colors = [
    #        'Reds', 'Greens', 'Blues', 'Purples', 'Oranges', 'Greys',
    #        'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
    #        'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']


	def multidim_cdf(self, pdf):
		cdf = pdf[...,::1].cumsum(1)[...,::1]
		for i in range(2,pdf.ndim+1):
			np.cumsum(cdf, axis=-i, out=cdf)
		return cdf

	def multi_joint_plot(self, table_vec, table_probs):
		self.colors = ['r', 'g', 'b']
		max_val = 10
		fig = plt.figure()
		ax1 = fig.add_subplot(111)
		dataframe = []
		a = 1

		for i in range(len(table_vec)):
			_x = []
			_y = []
			_z = []
			_color = []
			cdf_p = []
			pdf_table = np.zeros([max_val + 1, max_val + 1])
			cdf_table = np.zeros([max_val + 1, max_val + 1])


			for j in range(len(table_vec[i])):
				vec = table_vec[i]
				probs = table_probs[i]
				pdf_table[tuple(np.array(vec[j]))] = probs[j]


			
			for a in range(len(pdf_table)):
				for b in range(len(pdf_table)):
					_x.append(a)
					_y.append(b)
					_z.append(pdf_table[a][b])
					_color.append(self.colors[i])
			

			data = {'x' : _x, 'y': _y, 'z': _z, 'color': _color}
			dataframe.append(pd.DataFrame(data))

		df = pd.concat(dataframe) 
		df = df[df['z'] !=0]
		#print(df)
		#g = sns.JointGrid()
		x, y, z, colors = df['x'], df['y'], df['z'], df['color']
		#g = sns.JointGrid(data=df, x="x", y="y", hue='color', kind='hist', alpha = 0.5)
		#sns.scatterplot(x=x, y=y, ec=colors, fc="none", s=100, linewidth=1.5, ax=g.ax_joint)
		#sns.barplot(x=x, y = z ,color="red", data= df)
		#sns.barplot(x=z, y = y ,color="red", data= df)
		#g.plot_joint(sns.kdeplot, color="color", zorder=0, levels=6)
		#g.plot_marginals(sns.barplot, color="r")

		#g = sns.JointGrid(data=df, x=x, y=y, hue=colors)
		#g.plot(sns.scatterplot, sns.histplot)
		ax1.contourf(x, y, zdir='z', alpha=0.9)
		#sns.kdeplot(
    	#data=df, x=x, y=y, hue=colors, fill=True, alpha=0.5, weights=z
		#			)
		plt.show()

		return

	def heatmap_plot(self, table_vec, table_probs):
		max_val = 10
		self.colors = [
            'Reds', 'Greens', 'Blues', 'Purples', 'Oranges', 'Greys',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

		_x = []
		_y = []
		cdf_p = []
		pdf_table = np.zeros([max_val + 1, max_val + 1])

		
		for j in range(len(table_vec)):
			#vec = table_vec[j]
			#probs = table_probs[j]
			pdf_table[tuple(np.array(table_vec[j]))] = table_probs[j]

		plt.imshow(pdf_table, cmap=self.colors[0])
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
		print(table_vec)

		for i in range(len(table_vec)):
			#_x = []
			#_y = []
			#cdf_p = []
			
			vec = table_vec[i]
			probs = table_probs[i]

			for j in range(len(table_vec[i])):
				pdf_table[tuple(np.array(vec[j]))] = probs[j]			
		

		ax1.imshow(pdf_table, cmap='Blues')

		ax1.set_xlabel('objective 1')
		ax1.set_ylabel('objective 2')
		#ax1.set_zlabel('probability')

		ticks = [i for i in range(max_val + 1)]

		ax1.set_xticks(ticks)
		ax1.set_yticks(ticks)
		#ax1.set_zticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])

		ax1.axes.set_xlim(left=ticks[0], right=len(ticks) - 1) 
		ax1.axes.set_ylim(bottom=ticks[0], top=len(ticks) - 1) 
		#ax1.axes.set_zlim3d(bottom=0, top=1) 

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


	def exp_plot(self, table, _type):
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