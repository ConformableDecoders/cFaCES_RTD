import numpy as np
import matplotlib.pyplot as plt
from knn_dtw_training import KnnDtw
from sklearn.externals import joblib 
from datetime import datetime as dt
import os

import rt_capture_waveforms as capture 
import rt_classify_waveforms as classify
import rt_display_classification as display

import rt_util as tools

# read pickled dictionary created from rt_calibrate_motions to get labels
# labels = {0:'A', 1:'E', 2:'I', 3:'O', 4:'U', 5:'purse', 6:'open_mouth', 7:'twitch_small',
# 	            8:'twitch_medium', 9:'smile_small', 10:'smile_medium'}
labels_file = input ('Name of labels file (include the extension)? Press enter if none. ')
if '.pkl' in labels_file:
	labels = joblib.load('labels/' + labels_file)
	print('Successfully loaded ' + labels_file)
else: 
	labels = {0:'A', 1:'E', 2:'I', 3:'O', 4:'U', 5:'purse', 6:'open_mouth', 7:'twitch_small',
	            8:'twitch_medium', 9:'smile_small', 10:'smile_medium'}
	print ('Invalid labels file name. Reverting to default labels.')


print (labels)
trial_name = input('Name this trial: ')
if not os.path.exists('/home/pi/Desktop/Code/captured_motion_data/' + trial_name):
	os.mkdir('/home/pi/Desktop/Code/captured_motion_data/' + trial_name)
a = input('Press enter to continue.')
count = 0
cap = 400
trail = -100
window = 260

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Filter requirements.
order = 6
fs = 100       # sample rate, Hz
cutoff = 6  # desired cutoff frequency of the filter, Hz

while True:
	#Step 1: capture waveform through MCP3008 communication
	#capture_waveform(data_window_size (1/500)s, run_avg_size, num_run_avgs_stored, threshold_hi, threshold_lo, data_trail_size)
	waveforms = capture.capture_waveform(cap, 10, 10, 0.12, 0.07, 150)
	for waveform in waveforms:

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

		figname = input('Attempted motion: ')
		if figname != '\n':
			savename = 'captured_motion_data/' + trial_name + '/' + figname + '_' + dt.now().strftime('%d_%m_%y_%H_%M') + '.png'
			plt.savefig(savename, dpi=100)
		plt.show()

		wave_to_classify = np.array([wave_to_save1_filtered, wave_to_save2_filtered, wave_to_save3_filtered, wave_to_save4_filtered]).reshape((1,4,cap))
		label, proba, weighted_probs = classify.predict_classification(wave_to_classify, 3, 10, trial_name)
		display.display_prediction(labels, weighted_probs)

		# save waveform
		# write_file('4s_' + motion + '_' + str(file_ind + 1) + '_no_' + str(ind) + '.txt', waveform[1])
		tools.write_file_4elems('captured_motion_data/' + trial_name + '/4s_' + labels[label[0]] + '_no_' + str(count) + '.txt', wave_to_save1_filtered, wave_to_save2_filtered, wave_to_save3_filtered, wave_to_save4_filtered)
	count += 1
