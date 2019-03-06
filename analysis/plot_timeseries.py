import argparse
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description='Plot time series of data file at path.')
parser.add_argument('data_path', help='data path (ex: /Dug/data/noise/1.txt')
args = parser.parse_args()

# Load data
data = []

with open(args.data_path, 'r') as data_f:
    for row in data_f:        
        data.append(float(row))

data = np.array(data, dtype=float)
t = np.linspace(0, data.shape[0], num=data.shape[0])

# Plot timeseries
plt.plot(t, data)
plt.show()
