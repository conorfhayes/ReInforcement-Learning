import random
import numpy as np
import matplotlib.pyplot as plt


def main():
	#Stochastic Dominance Tests

	population1 = [-22]
	probs1 = [1]
	population2 = [-1000, -20]
	probs2 = [0.1, 0.9]
	population3 = [-1000, -12]
	probs3 = [0.15, 0.85]

	#population1 = [-16, -1000]
	#probs1 = [0.9, 0.1]
	#population2 = [-1000, -14]
	#probs2 = [0.19, 0.81]
	#population3 = [-1000, -6]
	#probs3 = [0.235, 0.765]

	population1 = [-22, -20, -1000, -12]
	probs1 = [0.34, 0.297, 0.0825, 0.2805]
	#print("Sum Pop 1", sum(probs1))
	population2 = [-1000, -16, -14, -6]
	probs2 = [0.18325, 0.297, 0.2673, 0.25245]
	#print("Sum Pop 2", sum(probs2))
	population3 = [-1000, -10, -8, 0]
	probs3 = [0.17755, 0.33, 0.25245, 0.24]

	#return


	#population1 = [-1000, -22, -20, -12]
	#probs1 = [0.09, 0.33, 0.3, 0.28]
	#print(sum(probs1))

	#population2 = [-1000, -16]
	#probs2 = [0.9, 0.1]

	#population3 = [-1000]
	#probs3 = [1]
	
	data_sorted1, prob_sorted1 = zip(*sorted(zip(population1, probs1)))
	data_sorted2, prob_sorted2 = zip(*sorted(zip(population2, probs2)))
	data_sorted3, prob_sorted3 = zip(*sorted(zip(population3, probs3)))

	p1_ = 0
	p1 = []
	for i in range(len(data_sorted1)):
		p1_ += prob_sorted1[i]
		p1.append(p1_)

	p2_ = 0
	p2 = []
	for i in range(len(data_sorted2)):
		p2_ += prob_sorted2[i]
		p2.append(p2_)

	p3_ = 0
	p3 = []
	for i in range(len(data_sorted3)):
		p3_ += prob_sorted3[i]
		p3.append(p3_)


	limits = [-1000, 0]

	def integrals(utility, probs, limits):

		increment = 1

		l1 = limits[0]
		l2 = limits[0] + increment
		
		area = []
		_area_ = 0
		
		a = 0
				
		while l1 < limits[1]:
			if a < len(utility) - 1 :
				if a == 0 and l1 < utility[a]:
					p = 0
				if l1 >= utility[a] and l1 < utility[a + 1]:
					p = probs[a]
				else:
					a += 1
					p = probs[a]

			elif a == 0 and l2 < utility[a]:
				p = 0
			elif a == 0 and l2 > utility[a]:
				p = probs[a]

			calc = (l1 * p) - (l2 * p)
			_area_ += calc

			area.append(abs(round(_area_, 2)))

			l1 += increment
			l2 += increment

		return area

	area1 = integrals(data_sorted1, p1, limits)
	area2 = integrals(data_sorted2, p2, limits)
	area3 = integrals(data_sorted3, p3, limits)	

	ssd10 = np.array(area2) - np.array(area1)
	ssd01 = np.array(area1) - np.array(area2)
	ssd20 = np.array(area3) - np.array(area1)
	ssd21 = np.array(area3) - np.array(area2)
	ssd02 = np.array(area1) - np.array(area3)
	ssd12 = np.array(area2) - np.array(area3)

	print("Distribution 0:", data_sorted1, p1)
	print("Distribution 1:", data_sorted2, p2)
	print("Distribution 2:", data_sorted3, p3)

	print("Distribution 1 - Distribution 0", min(ssd10))
	print("Distribution 0 - Distribution 1", min(ssd01))
	print("Distribution 2 - Distribution 0", min(ssd20))
	print("Distribution 2 - Distribution 1", min(ssd21))
	print("Distribution 0 - Distribution 2", min(ssd02))
	print("Distribution 1 - Distribution 2", min(ssd12))
	print("")

	samples1 = []
	samples2 = []
	samples3 = []
	
	n = 10000

	for j in range(20000):

		#for i in range(10):
		sample1 = np.random.choice(population1, n, p=probs1)
		sample2 = np.random.choice(population2, n, p=probs2)
		sample3 = np.random.choice(population3, n, p=probs3)
		#print(sample)
		val1 = sum(sample1)/ n
		samples1.append(val1)

		val2 = sum(sample2)/ n
		samples2.append(val2)

		val3 = sum(sample3)/ n
		samples3.append(val3)

		#print(samples1)
		#print(samples2)
		#print(samples3)
		#y = np.bincount(x)
		#ii = np.nonzero(y)[0]

		bs_sample1 = np.random.choice(samples1)
		bs_sample2 = np.random.choice(samples2)
		bs_sample3 = np.random.choice(samples3)

		print("**** Trial :", j, "****")
		print(bs_sample1)
		print(bs_sample2)
		print(bs_sample3)
		print(" ")

	unique1, counts1 = np.unique(samples1, return_counts=True)
	print(unique1, counts1)
	unique2, counts2 = np.unique(samples2, return_counts=True)
	print(unique2, counts2)
	unique3, counts3 = np.unique(samples3, return_counts=True)
	print(unique3, counts3)

	norm1 = np.random.normal(np.mean(population1), np.std(population1), size = 200)
	unique_norm, counts_norm = np.unique(norm1, return_counts=True)
	#print(unique_norm, counts_norm)
	#plt.bar(unique_norm, counts_norm)
	plt.bar(unique1, counts1, alpha=0.25)
	plt.bar(unique2, counts2, alpha=0.25)
	plt.bar(unique3, counts3, alpha=0.25)
	plt.title('Distributions of sample means')
	plt.xlabel("Utility")
	plt.ylabel("Frequency")
	#plt.legend(('label1', 'label2', 'label3'))
	plt.legend(('[1, -16]', '[1, -14] [0, -13]', '[1, -6] [0, -6]'),loc='upper center', shadow=False, ncol=3)
	plt.show()
	#plt.bar(unique2, counts2)
	#plt.show()
	#plt.show()

	for x in range(100):
		bs1 = np.random.choice(samples1)
		bs2 = np.random.choice(samples2)
		bs3 = np.random.choice(samples3)

		_all_ = [bs1, bs2, bs3]
		ans = max(_all_)
		index = _all_.index(max(_all_))
		print("^^^^ Roll :", x, "^^^^")
		print(_all_)
		print("Most Optimal Action :", index)
		print("Optimal Value :", ans)
		print(" ")


	"""

	
	range_ = [-1000, 0]
	fig=plt.figure()
	ax=fig.add_axes([0,0,1,1])
	ax.scatter(range_, samples1, color='r')
	ax.scatter(range_, samples2, color='b')
	ax.scatter(range_, samples3, color='g')
	#fig.set_xlabel('Grades Range')
	#fig.set_ylabel('Grades Scored')
	ax.set_title('scatter plot')
	plt.show()
	"""
	#print(samples2)
	"""
	check = []
	x = 100000
	c = 5
	for i in range(1,x):
		y = (2 * np.log(i))/ i
		calc = c * np.sqrt(y)
		check.append(calc)

	print(check)
	"""

if __name__ == '__main__':
	main()
