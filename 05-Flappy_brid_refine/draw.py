import math
import numpy as np
import matplotlib.pyplot as plt
epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 20000
epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(
    -1. * frame_idx / epsilon_decay)

x = np.arange(30000)
y = np.array(list(map(epsilon_by_frame,x)))


plt.plot(x,y)
plt.show()