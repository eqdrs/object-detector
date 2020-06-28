import numpy as np
import matplotlib.pyplot as plt

x = [0,0.2,0.3571428571,0.7272727273,0.8125,0.88,0.9158878505,0.9230769231,0.9237288136]

y = [1,0.9912280702,0.9561403509,0.9473684211,0.8596491228,0.5789473684,0.3421052632,0.2105263158,0.04385964912]

def integrate(x, y):
    area = np.trapz(y=y, x=x)
    return area


print(integrate(x, y))

plt.plot(x, y)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Curva Precision x Recall do Detector')
plt.grid(color='gray', linestyle='-', linewidth=0.15)
plt.show()