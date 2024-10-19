import matplotlib
matplotlib.use('TkAgg')

import pandas as pd
import matplotlib.pyplot as plt
# matplotlib.font_manager._rebuild()
# import matplotlib.font_manager
import numpy as np

plt.ion()

# Initial DataFrame setup
df = pd.DataFrame({'x': [], 'y': []})
# Create figure and axis
fig, ax = plt.subplots()
line, = ax.plot([], [], 'b')

# Set axis limits
ax.set_xlim(0, 100)
ax.set_ylim(-1, 1)

# Function to update the plot dynamically
def update_plot(df):
    line.set_xdata(df['x'])
    line.set_ydata(df['y'])
    fig.canvas.draw()  # Redraw the plot
    fig.canvas.flush_events()  # Process GUI events


# Simulate updating the DataFrame and plot
for i in range(100):
    # Simulate new data being added to the DataFrame
    new_data = {'x': [i], 'y': [np.sin(i / 10)]}
    df = df._append(pd.DataFrame(new_data), ignore_index=True)
    # Update the plot with the new data
    update_plot(df)

    # time.sleep(0.1)  # Slow down the loop for visibility

# Keep the final plot on the screen
# plt.ioff()  # Disable interactive mode
# plt.show()