import argparse
import matplotlib.pyplot as plt
import numpy as np

from scipy.fftpack import fft

parser = argparse.ArgumentParser(description='Plot FFT transform of data file at path.')
parser.add_argument('data_path', help='data path (ex: /Dug/data/noise/1.txt')
args = parser.parse_args()

# Load data
stimulus_data = []
baseline_data = []

with open(args.data_path, 'r') as data_f:
    for row in data_f:
        d_i, s_i = row.split(' ')
        d_i, s_i = float(d_i), float(s_i)
        if s_i:
            stimulus_data.append(d_i)
        else:
            baseline_data.append(d_i)

stimulus_data = np.array(stimulus_data, dtype=float)
baseline_data = np.array(baseline_data, dtype=float)

plot_stimulus = True
plot_baseline = True

if stimulus_data.shape[0] == 0:
    plot_stimulus= False
    stimulus_data = np.ones(baseline_data.shape[0])
if baseline_data.shape[0] == 0:
    plot_baseline = False
    baseline_data = np.ones(stimulus_data.shape[0])

# Apply fft and filter
def filtered_frequency_domain_data(signal, T=1.0/192.0):
    W = np.fft.fftfreq(int(signal.size/2) + 1, T)
    f_signal = np.abs(np.real(np.fft.rfft(signal)))
    f_signal[W < 7.5] = 0
    f_signal[W > 30] = 0
    f_signal /= max(f_signal)
    return f_signal, W
    
f_baseline_data, W_baseline = filtered_frequency_domain_data(baseline_data)
f_stimulus_data, W_stimulus = filtered_frequency_domain_data(stimulus_data)

# Plot frequency domain
if plot_baseline:
    plt.plot(W_baseline, f_baseline_data, label="Baseline", alpha=0.5, color='red')
if plot_stimulus:
    plt.plot(W_stimulus, f_stimulus_data, label="Stimulus", alpha=0.5, color='green')
plt.title("Spectral Density of Data")
plt.legend()
plt.xlabel("Frequency (Hz)")
plt.ylabel("Spectral Density")
plt.show()
