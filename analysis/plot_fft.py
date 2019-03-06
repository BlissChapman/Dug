import argparse
import matplotlib.pyplot as plt
import numpy as np

from scipy.fftpack import fft

parser = argparse.ArgumentParser(description='Plot FFT transform of data file at path.')
parser.add_argument('data_path', help='data path (ex: /Dug/data/noise/1.txt')
args = parser.parse_args()

# Load data
stimulus_data = []
no_stimulus_data = []

with open(args.data_path, 'r') as data_f:
    for row in data_f:
        d_i, s_i = row.split(' ')
        d_i, s_i = float(d_i), float(s_i)
        if s_i:
            stimulus_data.append(d_i)
        else:
            no_stimulus_data.append(d_i)

stimulus_data = np.array(stimulus_data, dtype=float)
no_stimulus_data = np.array(no_stimulus_data, dtype=float)

plot_stimulus = True
plot_baseline = True

if stimulus_data.shape[0] == 0:
    plot_stimulus= False
    stimulus_data = np.ones(no_stimulus_data.shape[0])
if no_stimulus_data.shape[0] == 0:
    plot_baseline = False
    no_stimulus_data = np.ones(stimulus_data.shape[0])

stimulus_data_N = stimulus_data.shape[0]
no_stimulus_data_N = no_stimulus_data.shape[0]
T = 1.0 / 260.0

#x = np.linspace(0.0, stimulus_data_N*T, stimulus_data_N)
#stimulus_data = np.sin(20.0 * 2.0*np.pi*x) + 0.5*np.sin(5.0 * 2.0*np.pi*x)

# Apply fft
stimulus_ps = np.abs(np.fft.fft(stimulus_data))**2
stimulus_ps /= max(stimulus_ps)
stimulus_freqs = np.fft.fftfreq(stimulus_data_N, T)
stimulus_idx = np.argsort(stimulus_freqs)

no_stimulus_ps = np.abs(np.fft.fft(no_stimulus_data))**2
no_stimulus_ps /= max(no_stimulus_ps)
no_stimulus_freqs = np.fft.fftfreq(no_stimulus_data_N, T)
no_stimulus_idx = np.argsort(no_stimulus_freqs)

# Plot frequency domain
if plot_baseline:
    plt.plot(no_stimulus_freqs[no_stimulus_idx], no_stimulus_ps[no_stimulus_idx], label="Baseline", alpha=0.5, color='red')
if plot_stimulus:
    plt.plot(stimulus_freqs[stimulus_idx], stimulus_ps[stimulus_idx], label="Stimulus", alpha=0.5, color='green')
plt.title("Power Spectral Density of Data")
plt.legend()
plt.xlabel("Frequency (Hz)")
plt.ylabel("Normalized Power Density")
plt.show()
