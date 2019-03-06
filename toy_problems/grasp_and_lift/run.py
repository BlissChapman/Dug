import mne
import numpy as np
import os
import pandas as pd

from imblearn.over_sampling import SMOTE, RandomOverSampler

from mne import create_info, find_events, Epochs, Evoked, EvokedArray
from mne.channels import read_montage
from mne.decoding import Vectorizer, get_coef, LinearModel, CSP
from mne.epochs import concatenate_epochs
from mne.io import RawArray
from mne.preprocessing import Xdawn

from matplotlib import pyplot as plt

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')

# Change to false once you are ready to train on the entire dataset. When true,
#  only a small representative sample of the training set will be used.
USE_DEBUG_TRAINING_DATASET = True

# Change to True when you are ready to cache the preprocessed data. This is set
#  to False by default because it is very easy to forget you are using cached data
#  and futile-y change the preprocessing code.
CACHE_ENABLED = False

# Sample frequency of the resampled dataset. The frequency of the dataset
#  when recorded was 500hz. If SFREQ < 500, the dataset will be resampled
#  to SFREQ.
SFREQ = 100

# Paths to the train and test datasets.
TRAIN_DATA_PATH = './data/train'
TEST_DATA_PATH = './data/test'

def create_mne_raw_object(fname):
    # Read EEG file
    data = pd.read_csv(fname)

    # Get channel names
    ch_names = list(data.columns[1:])

    # Read EEG standard montage from mne
    montage = read_montage('standard_1005', ch_names)

    # Events file
    ev_fname = fname.replace('_data', '_events')

    # Read event file
    events = pd.read_csv(ev_fname)
    events_names = events.columns[1:]
    events_data = np.array(events[events_names]).T

    # Concatenate event file and data
    data = np.concatenate( (np.array(data[ch_names]).T, events_data) )

    # Define channel type, the first is EEG, the last 6 are stimulations
    ch_type = ['eeg']*len(ch_names) + ['stim']*6

    ch_names.extend(events_names)

    # Create and populate MNE info structure
    info = create_info(ch_names, sfreq=500.0, ch_types=ch_type, montage=montage)

    # Create raw object
    raw = RawArray(data, info, verbose=False)

    return raw

def unique_rows(a):
    order = np.lexsort(a.T)
    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1) 
    return a[ui]

def load_subject_series_epochs(data_path, subject_range, series_range, tmin=-0.2, tmax=0.5, baseline=(None, 0), stim_channel=None):
    all_epochs_list = []

    for subject in subject_range:
        subject_fname = data_path + '/subj{0}'.format(subject)

        for series in series_range:
            subject_series_fname = subject_fname + '_series{0}_data.csv'.format(series)
            subject_series = create_mne_raw_object(subject_series_fname)
            subject_series_events = find_events(subject_series, stim_channel=stim_channel, verbose=False)
            epochs = Epochs(subject_series, subject_series_events, tmin=tmin, tmax=tmax, baseline=baseline, verbose=False)
            all_epochs_list.append(epochs)
            
    all_epochs = concatenate_epochs(all_epochs_list)
    return all_epochs

def load_subject_series_data(data_path, subject_range, series_range):
    data = None

    for subject in subject_range:
        subject_fname = data_path + '/subj{0}'.format(subject)

        for series in series_range:
            subject_series_fname = subject_fname + '_series{0}_data.csv'.format(series)
            subject_series = create_mne_raw_object(subject_series_fname)

            if data:
                data.append(subject_series)
            else:
                data = subject_series

    return data

train_subject_range = range(1, 12) if not USE_DEBUG_TRAINING_DATASET else range(1, 2)
train_series_range = range(1, 9)

test_subject_range = range(12, 13)
test_series_range = range(1, 9)

print("TRAIN SUBJECT RANGE: {0}".format(train_subject_range))

# Load all data
all_train_data = load_subject_series_data(data_path=TRAIN_DATA_PATH,
                                          subject_range=train_subject_range,
                                          series_range=train_series_range)
all_test_data = load_subject_series_data(data_path=TRAIN_DATA_PATH,
                                         subject_range=test_subject_range,
                                         series_range=test_series_range)

# Downsample train and test data.
all_train_data = all_train_data.resample(SFREQ, npad='auto')
all_test_data = all_test_data.resample(SFREQ, npad='auto')

# Split train data into data and labels
x_train = all_train_data.copy().pick_types(eeg=True).filter(7, 35,
                                                            method='iir',
                                                            n_jobs=-1,
                                                            verbose=False).to_data_frame().values
y_train = all_train_data.copy().pick_types(stim=True).to_data_frame().values
y_train = y_train > 0

# Split test data into data and labels
x_test = all_test_data.copy().pick_types(eeg=True).filter(7, 35,
                                                          method='iir',
                                                          n_jobs=-1,
                                                          verbose=False).to_data_frame().values
y_test = all_test_data.copy().pick_types(stim=True).to_data_frame().values
y_test = y_test > 0

print("\nDATA SHAPE: {0}".format(x_train.shape))
print("LABEL SHAPE: {0}".format(y_train.shape))
print("\nSAMPLE DATA POINT: {0}".format(x_train[5]))

# Discard references to original data for memory performance
all_train_data = None
all_test_data = None

# Extract 'HandStart' labels
class_type = 'HandStart'
y_train_class_type = y_train[:, 0]
y_test_class_type = y_test[:, 0]

y_train = None
y_test = None

# Normalize data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print("\nSAMPLE DATA POINT: {0}".format(x_train[5]))

# Concatenate time points between t_start and t_end
t_start = -0.2
t_end = 0.3

new_x_train = x_train.copy()
new_x_test = x_test.copy()

for t_i in range(int(t_start*SFREQ), int(t_end*SFREQ)+1, 1):
    if t_i == 0:
        continue
    new_x_train = np.hstack((new_x_train, np.roll(x_train, -t_i, axis=0)))
    new_x_test = np.hstack((new_x_test, np.roll(x_test, -t_i, axis=0)))

x_train = new_x_train
x_test = new_x_test

print("\nDATA SHAPE: {0}".format(x_train.shape))
print("\nSAMPLE DATA POINT: {0}".format(x_train[5]))

# Rebalance data
print("% POSITIVE LABELS IN TRAIN SET BEFORE REBALANCE: {0:.2f}%".format(100*np.sum(y_train_class_type)/y_train_class_type.shape[0]))

num_negative_samples = y_train_class_type.shape[0] - np.sum(y_train_class_type)
sm = SMOTE(random_state=12, sampling_strategy={0:num_negative_samples, 1:1*num_negative_samples}, n_jobs=4)
ros = RandomOverSampler(random_state=12, sampling_strategy={0:num_negative_samples, 1:1*num_negative_samples})

x_train, y_train_class_type = sm.fit_resample(x_train, y_train_class_type)

sm = None
ros = None

print("% POSITIVE LABELS IN TRAIN SET AFTER REBALANCE: {0:.2f}%".format(100*np.sum(y_train_class_type)/y_train_class_type.shape[0]))

# Fit model on class_type labels
model = LogisticRegression(solver='sag', n_jobs=-1, verbose=True)
model.fit(x_train, y_train_class_type)

# Make predictions on both train, train (rebalanced), and test set
y_train_class_type_pred = model.predict(x_train)
y_test_class_type_pred = model.predict(x_test)

print("\n============= [TRAIN METRICS] =============\n")
print(classification_report(y_train_class_type, y_train_class_type_pred))

print("\n============= [TEST METRICS] =============\n")
print(classification_report(y_test_class_type, y_test_class_type_pred))

# Fit model on class_type labels
model = MLPClassifier(hidden_layer_sizes=(100,), verbose=True)
model.fit(x_train, y_train_class_type)

# Make predictions on both train, train (rebalanced), and test set
y_train_class_type_pred = model.predict(x_train)
y_test_class_type_pred = model.predict(x_test)

print("\n============= [TRAIN METRICS] =============\n")
print(classification_report(y_train_class_type, y_train_class_type_pred))

print("\n============= [TEST METRICS] =============\n")
print(classification_report(y_test_class_type, y_test_class_type_pred))