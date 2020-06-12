import random
import numpy as np
import matplotlib.pyplot as plt

def main():
	#Stochastic Dominnace Tests

	population1 = [-22]
	probs1 = [1]
	population2 = [-1000, -20]
	probs2 = [0.1, 0.9]
	population3 = [-1000, -12]
	probs3 = [0.15, 0.85]

	
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

	print(area3)

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

if __name__ == '__main__':
	main()