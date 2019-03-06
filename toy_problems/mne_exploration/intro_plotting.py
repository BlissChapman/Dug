import mne

from mne.datasets import sample
from matplotlib import pyplot as plt

# ========== RAW DATA ==========
# Read the raw data
data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
raw = mne.io.read_raw_fif(raw_fname, preload=True)

# Useful:
# raw.ch_names
# raw.info

# ========== PREPROCESS DATA ==========
# Mark bad channels
raw.info['bads'] = []

data = raw.copy().pick_types(meg=False, eeg=True)

# Plot the processed data
fig = data.plot(highpass=1.0, lowpass=70.0)
plt.show(fig)

fig = data.plot_psd()
plt.show(fig)

fig = data.plot_psd_topo()
plt.show(fig)

fig = data.plot_sensors()
plt.show(fig)
