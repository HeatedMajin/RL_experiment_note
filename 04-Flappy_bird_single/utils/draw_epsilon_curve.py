import math
import numpy as np
import matplotlib.pyplot as plt
epsilon_start = 1.0
epsilon_final = 0.1

x = np.array([1,20000])
y = np.array([1.0,0.1])

x_ticks = np.arange(0,20000+1,2000)
y_ticks = np.arange(0.1,1,0.1)
plt.plot(x,y)
plt.xticks(x_ticks)
plt.yticks(y_ticks)
plt.show()