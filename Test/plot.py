import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
plt.rcParams['font.family'] = 'Times New Roman'
list = [1, 2, 3, 4]
plt.plot(list)

plt.xlabel('frames')
plt.ylabel('avg. density error')
plt.show()