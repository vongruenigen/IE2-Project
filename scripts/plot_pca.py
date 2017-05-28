import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import csv
from matplotlib.pyplot import cm 
from numpy import array

CLUSTER_PATH = "C:/Users/MWeil/Documents/GitHub/IE2-Project/data/pca/"
CLUSTERS = 7

def getColumn(filename, column):
    results = csv.reader(open(filename), delimiter=";")
    return [result[column] for result in results]

centerx = []
centery = []
colors = iter(cm.rainbow(np.linspace(0, 1, CLUSTERS)))

for n in range(0, CLUSTERS):
	filename = CLUSTER_PATH + "onehot_cluster_" + str(n) + "/part-00000"
	x = getColumn(filename,0)
	y = getColumn(filename,1)
	plt.figure("x/y")
	plt.xlabel("x")
	plt.ylabel("y")
	color = list(next(colors))[0:3]
	plt.scatter(x,y,c=color,marker='.')
	centerx.append(x[-1])
	centery.append(y[-1])
plt.scatter(centerx,centery,c='red',marker='x',s=30)
for n in range(0, CLUSTERS):
	plt.annotate("center" + str(n), xy=(centerx[n], centery[n]), xytext=(-2, -2 -n/CLUSTERS),
            arrowprops=dict(facecolor='yellow', shrink=0.05),
            )
plt.legend()
plt.show()
