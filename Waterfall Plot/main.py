import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Path to CSV file
csv_file_path = 'C:\\Users\\Emil\\PycharmProjects\\CapstoneKerasML\\Testing\\RandomizedData.csv'

# Load the data from CSV into a numpy array
psd_data = pd.read_csv(csv_file_path, header=None).values

# Plotting
plt.figure(figsize=(10, 6))
plt.imshow(np.log10(psd_data + 1), aspect='auto', origin='lower',
           extent=[0, 300, 0, 2048], cmap='viridis')  # 'viridis' or any other colormap
plt.colorbar(label='Log10(PSD)')
plt.xlabel('Column Index')
plt.ylabel('Frequency Bin')
plt.title('Waterfall Plot of PSD Data')
plt.show()


