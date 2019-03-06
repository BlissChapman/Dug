import argparse
import numpy as np
import pickle

from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


parser = argparse.ArgumentParser(description='Train, evaluate, and save classifier on data at specified path.')
parser.add_argument('data_path', help='data path (ex: /Dug/data/noise/1.txt)')
parser.add_argument('model_type', help='the type of model to train (supports: SVC and Logistic Regression)')
parser.add_argument('output_path', help='output path (ex: /Dug/src/pi/models/LogisticRegression)')
args = parser.parse_args()

# ======================================================================
# Load Data
# ======================================================================
x = [] # EEG waveform
y = [] # Labels (stimuli or not)

with open(args.data_path, 'r') as data_f:
	for row in data_f:
		d_i, s_i = row.split(' ')
		x.append(float(d_i))
		y.append(float(s_i))

x = np.reshape(np.array(x, dtype=float), (-1, 1))
y = np.array(y, dtype=float)

# NOTE TO FUTURE READERS:
# Our analog circuit attenuates signals with frequency greater than 31 hz
#  and less than 7 hz and amplifies the signal with a gain of __.
# Data reaches Russell (running on the Pi) at ~192hz.
def SAMPLES_TO_SECONDS(n):
	return n*52.0164690018/10000.0 # ~192hz
	
def SECONDS_TO_SAMPLES(t):
	return t*10000.0/52.0164690018 # ~192hz
	
# ======================================================================
# Preprocess Data
# ======================================================================
# Transform dataset so that what we are predicting is what happened 
#  0.5 seconds ago and NOT what is happening RIGHT now?

	
# Concatenate time points between t_start and t_end
t_start = -0.2
t_end = 0.5

new_x = x.copy()

for t_i in range(int(SECONDS_TO_SAMPLES(t_start)), int(SECONDS_TO_SAMPLES(t_end))+1, 1):
    if t_i == 0:
        continue
    new_x = np.hstack((new_x, np.roll(x, -t_i, axis=0)))
    
x = new_x

# Apply Fast Fourier Transform to every set of time points
for i in range(0, x.shape[0]):
	x[i] = np.real(np.fft.fft(x[i]))

# ======================================================================
# Train Model
# ======================================================================

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# Normalize data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Rebalance
num_samples_max_class = max(y_train.shape[0] - int(np.sum(y_train)), int(np.sum(y_train)))
sm = SMOTE(random_state=42, sampling_strategy={0:num_samples_max_class, 1:num_samples_max_class})
x_train, y_train = sm.fit_resample(x_train, y_train)

# Fit model
if args.model_type == "SVC":
	model = SVC(verbose=True)
elif args.model_type == "LR":
	model = LogisticRegression(solver='sag', n_jobs=-1, verbose=True)
else:
	print("Unknown model type '{0}' specified. View the params help guide to see supported model types.")
	exit(1)
	
model.fit(x_train, y_train)

# ======================================================================
# Evaluate Model
# ======================================================================

# Make predictions
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

print("\n============= [TRAIN METRICS] =============\n")
print(classification_report(y_train, y_train_pred))

print("\n============= [TEST METRICS] =============\n")
print(classification_report(y_test, y_test_pred))

# ======================================================================
# Save Model
# ======================================================================

with open(args.output_path+'scaler.pkl', 'wb') as scaler_f:
        pickle.dump(scaler, scaler_f)
with open(args.output_path+'model.pkl', 'wb') as model_f:        
        pickle.dump(model, model_f)
