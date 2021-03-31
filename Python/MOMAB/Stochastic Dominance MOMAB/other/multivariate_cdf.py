import numpy as np
import matplotlib.pyplot as plt

# example numbers
# [1, 1] 0.1
# [1, 5] 0.2
# [2, 3] 0.5
# [1, 2] 0.2

# cdf

vectors = [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], 
		[1,	0], [1,	1], [1,	2], [1,	3], [1,	4], [1,	5], 
		[2,	0], [2,	1], [2,	2], [2,	3], [2,	4], [2,	5], 
		[3,	0], [3,	1], [3,	2], [3,	3], [3,	4], [3,	5], 
		[4,	0], [4,	1], [4,	2], [4,	3], [4,	4], [4,	5], 
		[5,	0], [5,	1], [5,	2], [5,	3], [5,	4], [5,	5]]
probs = [0, 0, 0, 0, 0, 0, 
		0, 0.1, 0.3, 0.3, 0.3, 0.5, 
		0, 0.1, 0.3, 0.8, 0.8, 0.8, 
		0, 0.1, 0.3, 0.8, 0.8, 1, 
		0, 0.1, 0.3, 0.8, 0.8, 1, 
		0, 0.1, 0.3, 0.8, 0.8, 1]


# setup the figure and axes
fig = plt.figure(figsize=(8, 3))
ax1 = fig.add_subplot(111, projection='3d')

#vectors = [[1,1], [1,5], [2,3], [1,2]]
#probs = [0.1, 0.2, 0.5, 0.2]
_x = []
_y = []

# fake data
for i in vectors:
	for j in range(len(i)):
		if j == 0:
			_x.append(i[j])
		if j == 1:			
			_y.append(i[j])

#_x = [1,2,3,4,5,6,7,8,9,10]
#_y = [5,6,7,8,2,5,6,3,7,2]
_z = np.zeros(len(probs))
print(_z)
#_xx, _yy = np.meshgrid(_x, _y)
#x, y = _xx.ravel(), _yy.ravel()

dx = np.ones(len(_x))
dy = np.ones(len(_y))
dz = probs

#ax1.bar3d(x, y, bottom, width, depth, top, shade=True)

#ax1.bar3d(_x, _y, _z, 1, 1, dz, shade = True)
ax1.plot_trisurf(_x, _y, dz, cmap='twilight_shifted')
#ax1.scatter(_x, _y, dz, c=dz, cmap='BrBG', linewidth=1)

ax1.set_xlabel('objective 1')
ax1.set_ylabel('objective 2')
ax1.set_zlabel('probability')

ax1.set_xticks([0,1,2,3,4,5])
ax1.set_yticks([0,1,2,3,4,5])
ax1.set_zticks([0.1,0.2,0.3,0.4,0.5, 0.6, 0.7, 0.8, 0.9, 1])

ax1.axes.set_xlim3d(left=0, right=5) 
ax1.axes.set_ylim3d(bottom=0, top=5) 
ax1.axes.set_zlim3d(bottom=0, top=1) 

plt.show()
