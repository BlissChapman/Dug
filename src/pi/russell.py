import argparse
import RPi.GPIO as GPIO
import numpy as np
import os
import pickle
import serial

parser = argparse.ArgumentParser(description='Stream data from specified serial port, classify it with the model at path, play audio blurbs when stimulus is predicted.')
parser.add_argument('port', help='serial port (ex: /dev/ttyACM0)')
parser.add_argument('model_path', help='path to model (ex: /Dug/src/pi/models/LogisticRegression)')
args = parser.parse_args()

# ======================================================================
# Status Indicator
# ======================================================================
STATUS_PIN = 26
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(STATUS_PIN, GPIO.OUT)

def set_status_indicator(status):
	print(status)
	GPIO.output(STATUS_PIN, GPIO.HIGH if status else GPIO.LOW)

# ======================================================================
# Classification
# ======================================================================

scaler = None
model = None

with open(args.model_path+'scaler.pkl', 'rb') as scaler_f:
        scaler = pickle.load(scaler_f)
with open(args.model_path+'model.pkl', 'rb') as model_f:
        model = pickle.load(model_f)

def classify_data_window(x):
	x = np.reshape(np.real(np.fft.fft(x)), (1, -1))
	x = scaler.transform(x)
	stimulus_present_prediction = model.predict(x)[0]
	print("STIMULUS PRESENT: {0}".format(stimulus_present_prediction))
	return stimulus_present_prediction

# ======================================================================
# Setup sound system
# ======================================================================

RESOURCE_PTH = '~/Dug/resources/sounds/'
SOUND_ALMA_INTRO     = RESOURCE_PTH + 'alma_intro.mp3'
SOUND_ALMA_TREAT     = RESOURCE_PTH + 'alma_treat.mp3'
SOUND_ALMA_DELICIOUS = RESOURCE_PTH + 'alma_delicious.mp3'
SOUND_ALMA_BAD       = RESOURCE_PTH + 'alma_bad.mp3'
SOUND_ALMA_UNSURE    = RESOURCE_PTH + 'alma_unsure.mp3'

def play_sound(resource):
	os.system('omxplayer -o local {0}'.format(resource))

# Sound check
play_sound(SOUND_ALMA_TREAT)

# ======================================================================
# Data window
# ======================================================================

# Setup sliding window that holds latest N time points
def SAMPLES_TO_SECONDS(n):
	return n*52.0164690018/10000.0 # ~192hz

def SECONDS_TO_SAMPLES(t):
	return t*10000.0/52.0164690018 # ~192hz

# 0.2 seconds pre-stimulus and 0.5 seconds post-stimulus is a reasonable
# choice for the size of a sliding window used in classification
N = SECONDS_TO_SAMPLES(0.2+0.5)+1

# How many samples should we ignore? This is set to a non-zero value
# after a treat classification is made.
IGNORE_DATA_N = 0

# Data structure to hold the sliding window of data.
data = []

# ======================================================================
# Main Loop
# ======================================================================

status_indicator = True

# Open serial port
ser = serial.Serial(args.port, 9600)

# Main loop
while True:

	# Check if there is data to read
	if ser.in_waiting == 0:
		continue

	# Read data from serial port
	data_i = ser.readline()
	try:
		data_i = int(data_i)
	except ValueError:
		continue

	# If there are data points left to ignore, ignore this data point!
	if IGNORE_DATA_N > 0:
		IGNORE_DATA_N -= 1
		continue

	# Update sliding window with new data
	data.append(data_i)
	if len(data) > N:
		data.pop(0) # Remove the oldest data from the sliding window
	else:
		# Wait until a full window is available before attempting
		#  classification
		continue
		
	# Update status indicator
	set_status_indicator(status_indicator)
	status_indicator = not status_indicator

	# Classification
	if classify_data_window(np.array(data)):
		play_sound(SOUND_ALMA_TREAT)
		data = []                              # Reset data window
		IGNORE_DATA_N = SECONDS_TO_SAMPLES(30) # Ignore next 30s of data
	else:
		# Performance hack:
		data = []
		IGNORE_DATA_N = SECONDS_TO_SAMPLES(1)  # Ignore next 1s of data
