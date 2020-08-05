from sklearn.externals import joblib 
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz
import csv
from datetime import datetime as dt
import os

import rt_capture_waveforms as capture 
import rt_classify_waveforms as classify

import rt_util as tools

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def capture_calibration_waveform(motion, labels, label_count, count, trial_name):
	cap = 400
	trail = -100
	window = 260
	#Step 1: capture waveform through MCP3008 communication
	#capture_waveform(data_window_size (1/100)s, run_avg_size, num_run_avgs_stored, threshold_hi, threshold_lo, data_trail_size)
	waveforms = capture.capture_waveform(cap, 10, 10, 0.12, 0.07, 150)
	waveform = waveforms[0]

	# Filter requirements.
	order = 6
	fs = 100       # sample rate, Hz
	cutoff = 6  # desired cutoff frequency of the filter, Hz

	# center waveform
	wave1 = tools.center_waveform(waveform[1], trail, window, cap)
	wave2 = tools.center_waveform(waveform[2], trail, window, cap)
	wave3 = tools.center_waveform(waveform[3], trail, window, cap)
	wave4 = tools.center_waveform(waveform[4], trail, window, cap)

	# remove dc offset
	wave_to_save1 = [val-sum(wave1)/len(wave1) for val in wave1]
	wave_to_save2 = [val-sum(wave2)/len(wave2) for val in wave2]
	wave_to_save3 = [val-sum(wave3)/len(wave3) for val in wave3]
	wave_to_save4 = [val-sum(wave4)/len(wave4) for val in wave4]

	wave_to_save1_filtered = butter_lowpass_filter(wave_to_save1, cutoff, fs, order)
	wave_to_save2_filtered = butter_lowpass_filter(wave_to_save2, cutoff, fs, order)
	wave_to_save3_filtered = butter_lowpass_filter(wave_to_save3, cutoff, fs, order)
	wave_to_save4_filtered = butter_lowpass_filter(wave_to_save4, cutoff, fs, order)

	# plot waveform
	# plt.plot(waveform[1], linewidth=1)
	plt.figure(figsize=(6,12))
	plt.subplot(4, 1, 1)
	plt.plot(wave_to_save1, linewidth=1)
	plt.plot(wave_to_save1_filtered)
	plt.ylabel('Element 1')

	plt.subplot(4, 1, 2)
	plt.plot(wave_to_save2, linewidth=1)
	plt.plot(wave_to_save2_filtered)
	plt.ylabel('Element 2')

	plt.subplot(4, 1, 3)
	plt.plot(wave_to_save3, linewidth=1)
	plt.plot(wave_to_save3_filtered)
	plt.ylabel('Element 3')

	plt.subplot(4, 1, 4)
	plt.plot(wave_to_save4, linewidth=1)
	plt.plot(wave_to_save4_filtered)
	plt.ylabel('Element 4')
	plt.xlabel('Time (1/100 s)')

	savename = 'training_data/' + trial_name + '/' + motion + '_no_' + str(count) + '_' + dt.now().strftime('%d_%m_%y_%H_%M') + '.png'
	# plt.savefig(savename, dpi=150)
	plt.show()

	valid = input('Is this a valid waveform? y for yes, anything else for no. ')
	is_valid = True if valid == 'y' or valid == 'Y' else False

	if is_valid:
		# save waveforms
		tools.write_file_4elems('training_data/' + trial_name + '/4s_' + labels[label_count] + '_no_' + str(count) + '.txt', wave_to_save1_filtered, wave_to_save2_filtered, wave_to_save3_filtered, wave_to_save4_filtered)
		count += 1

	again = input ('Capture another waveform? y for yes, anything else for no. ')
	run_again = True if again == 'y' or again == 'Y' else False

	if run_again:
		capture_calibration_waveform(motion, labels, label_count, count, trial_name)

def calibrate(labels, label_count, labels_name, trial_name):
	# pickle labels dictionary
	count = 0
	motion = input('What motion do you want to calibrate? ')
	labels[label_count] = motion
	capture_calibration_waveform(motion, labels, label_count, count, trial_name)

	label_count += 1
	again = input ('Calibrate another motion? y for yes, anything else for no. ')
	run_again = True if again == 'y' or again == 'Y' else False

	if run_again:
		calibrate(labels, label_count, labels_name, trial_name)
	else:
		# pickle the labels file
		joblib.dump(labels, 'labels/' + labels_name + '.pkl')

cal_or_train = input('Type c for calibrate or t for data reformat and training: ')

if cal_or_train == 'c':
	existing_label = input('Existing labels file (with extension): Or press Enter to create a new labels file. ')
	labels = joblib.load('labels/' + existing_label) if '.pkl' in existing_label else {}
	label_count = len(labels.keys()) if '.pkl' in existing_label else 0
	labels_name = input('Desired labels file name (no extension): ')
	trial_name = input('Name this trial: ')
	if not os.path.exists('/home/pi/Desktop/Code/training_data/' + trial_name):
		os.mkdir('/home/pi/Desktop/Code/training_data/' + trial_name)
	calibrate(labels, label_count, labels_name, trial_name)

if cal_or_train == 't':
	labels_name = input('Existing labels file (no extension): ')
	trial_name = input('Name of trial: ')

labels = joblib.load('labels/' + labels_name  + '.pkl')
print ('Labels in calibration script', labels)

reform = input ('Reformat the data into training and testing files now? y for yes, anything else for no. ')
reformat = True if reform == 'y' or reform == 'Y' else False

if reformat:
	# reformat data to match 
	import rt_reformat_calibration_data

trainit = input ('Train the classifier now? y for yes, anything else for no. ')
train = True if trainit == 'y' or trainit == 'Y' else False

if train:
	nn = int(input('How many nearest neighbors? '))
	ww = int(input('Search radius? '))
	classify.train_classifier(nn, ww, labels, trial_name)