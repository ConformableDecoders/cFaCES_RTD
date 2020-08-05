import numpy as np
import csv
import glob
import random
import matplotlib.pyplot as plt
import time
from gpiozero import MCP3008

elem1 = MCP3008(0)
elem2 = MCP3008(1)
elem3 = MCP3008(2)
elem4 = MCP3008(3)

# read the stream of data 

# store 4 seconds of data at a time

# calculate the absolute value of the running average of the data stream, 
# from the most recent data in window of size run_avg_wind_size

# store the last 100 running averages (0.1 seconds) continuously.

# once the average of the last 100 running averages reaches above a threshold,
# enter a loop which does the following:
	# once the average of the last 100 running averages reaches below a threshold, 
	# save this data stream of 4 seconds as a waveform

# take this last saved data stream and feed it into the KNN-DTW model for classification

def capture_waveform(data_window_size, run_avg_size, num_run_avgs_stored, 
				threshold_hi, threshold_lo, data_trail_size):

	# set up a stream variable that's constantly saving the last data_window_size data points
	stream1 = [0 for i in range(data_window_size)]
	stream2 = [0 for i in range(data_window_size)]
	stream3 = [0 for i in range(data_window_size)]
	stream4 = [0 for i in range(data_window_size)]
	# set up variable to save count when a potential waveform has ended
	potential_waveform_end_count = 10000000000
	midvalue1 = elem1.value
	midvalue2 = elem2.value
	midvalue3 = elem3.value
	midvalue4 = elem4.value
	# set up a list to store all the potential waveforms
	# potential_waveforms = []
	# set up a list to store running averages of length run_avg_size
	running_avgs1 = [0 for i in range(num_run_avgs_stored)]
	running_avgs2 = [0 for i in range(num_run_avgs_stored)]
	running_avgs3 = [0 for i in range(num_run_avgs_stored)]
	running_avgs4 = [0 for i in range(num_run_avgs_stored)]
	# set up a count to keep track of location in data stream
	count = 0

	# wait for a potential waveform for 10 minutes before giving up
	for i in range (100*60*10):
		# start timer
		start = time.time()

		# TO DO: Map voltage value to real mV scale (not centered at 0.5 but at 0)
		# lambda x: (x-midvalue)*3.3*1000
		# print (voltage.value)
		data_point1 = (elem1.value - midvalue1)*3.3
		data_point2 = (elem2.value - midvalue2)*3.3
		data_point3 = (elem3.value - midvalue3)*3.3
		data_point4 = (elem4.value - midvalue4)*3.3
		print (data_point1,'\t',data_point2,'\t',data_point3,'\t',data_point4)

		# calculate DC offset by averaging stream
		dc_offset1 = sum(stream1[:(int(data_window_size/2))])/len(stream1[:(int(data_window_size/2))])
		dc_offset2 = sum(stream2[:(int(data_window_size/2))])/len(stream2[:(int(data_window_size/2))])
		dc_offset3 = sum(stream3[:(int(data_window_size/2))])/len(stream3[:(int(data_window_size/2))])
		dc_offset4 = sum(stream4[:(int(data_window_size/2))])/len(stream4[:(int(data_window_size/2))])
		# update data stream held in memory as new data gets read
		stream1 = stream1[-(data_window_size-1):] + [data_point1 - dc_offset1]
		stream2 = stream2[-(data_window_size-1):] + [data_point2 - dc_offset2]
		stream3 = stream3[-(data_window_size-1):] + [data_point3 - dc_offset3]
		stream4 = stream4[-(data_window_size-1):] + [data_point4 - dc_offset4]
		# calculate running average of length run_avg_size
		# avg = sum(np.abs(stream[-1*run_avg_size:]))/float(run_avg_size)
		avg1 = sum(stream1[-1*run_avg_size:])/float(run_avg_size)
		avg2 = sum(stream2[-1*run_avg_size:])/float(run_avg_size)
		avg3 = sum(stream3[-1*run_avg_size:])/float(run_avg_size)
		avg4 = sum(stream4[-1*run_avg_size:])/float(run_avg_size)
		# update stream of running averages stored in memory
		running_avgs1 = running_avgs1[-(num_run_avgs_stored-1):] + [avg1]
		running_avgs2 = running_avgs2[-(num_run_avgs_stored-1):] + [avg2]
		running_avgs3 = running_avgs3[-(num_run_avgs_stored-1):] + [avg3]
		running_avgs4 = running_avgs4[-(num_run_avgs_stored-1):] + [avg4]
		# calculate the average of averages saved so far
		avg_of_avgs1 = np.abs(sum(running_avgs1))/len(running_avgs1)
		avg_of_avgs2 = np.abs(sum(running_avgs2))/len(running_avgs2)
		avg_of_avgs3 = np.abs(sum(running_avgs3))/len(running_avgs3)
		avg_of_avgs4 = np.abs(sum(running_avgs4))/len(running_avgs4)

		# any way to save the waveform centered at the right spot in the window?
		# need to track potential waveform start as well as end

		# sometimes I'm getting the same signal twice, but one is shifted.
		# for PoC, just ignore those that are not classifiable with high enough accuracy
		# (this means I need to add another parameter to the output of the knn_dtw prediction)

		if avg_of_avgs1 > threshold_hi or avg_of_avgs2 > threshold_hi or avg_of_avgs3 > threshold_hi or avg_of_avgs4 > threshold_hi:
			potential_waveform_end_count = count

		if avg_of_avgs1 < threshold_lo and avg_of_avgs2 < threshold_lo and avg_of_avgs3 < threshold_lo and avg_of_avgs4 < threshold_lo and count > potential_waveform_end_count + data_trail_size:
			# if potential_waveform_end_count not in [x[0] for x in potential_waveforms]:
			return [(potential_waveform_end_count, stream1, stream2, stream3, stream4)]

		count += 1

		while (time.time() - start < 9.5*10**-3): #500 Hz; 9.5 for 100 Hz
			pass
			# do nothing

