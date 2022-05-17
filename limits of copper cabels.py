import matplotlib.pyplot as plt
import numpy as np
s1 = [.5, .75, 1, 1.5, 2, 2.5 , 4, 6, 10, 16, 25, 35]
s2 = [11, 15, 17, 23, 26, 30, 41, 50, 80, 100, 140, 170]
s1, s2 = s2, s1
p, v = np.polyfit(np.log(s1), np.log(s2), deg = 1, cov = True)
x = np.arange(min(s1), max(s1), (max(s1)-min(s1))/1000)
plt.plot(s1, s2, 'bs')
plt.plot(x, p[1]*x**p[0])
plt.show()
