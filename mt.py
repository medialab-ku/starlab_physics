import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# matplotlib.font_manager._rebuild()
# import matplotlib.font_manager
import numpy as np



# print(matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf'))

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
# Sample data
x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 4, 9, 16, 25])

# Create a plot

plt.plot(x, label=r'$\mathrm{y}=\mathrm{x}^2$')
plt.plot(y, label='y')

# Add a title and labels
plt.title('Basic Line Plot', fontsize=20)
plt.xlabel('X axis', fontsize=14)
plt.ylabel('Y axis', fontsize=14)
# Show the plot
plt.legend(fontsize=20, loc='upper left')
plt.show()