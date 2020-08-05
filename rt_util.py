import numpy as np
import csv

def write_file(file_name, array_to_write):
	'''
	Saves captured waveforms, which can be used for calibration or classification

	parameters: 
		file_name: name of file to write
		array_to_write: 1D array of data points, assumes you know the data rate
	'''

	file = open(file_name, 'w')
	for item in array_to_write:
		file.write(str(item))
		file.write('\t')
		# file.write('\n')
	file.close()

def write_file_4elems(file_name, arr1, arr2, arr3, arr4):
	'''
	Saves captured waveforms, which can be used for calibration or classification

	parameters: 
		file_name: name of file to write
		array_to_write: 1D array of data points, assumes you know the data rate
	'''

	arrs = [arr1, arr2, arr3, arr4]

	file = open(file_name, 'w')
	for i in range(len(arr1)):
		for arr_i in range(len(arrs)):
			file.write(str(arrs[arr_i][i]))
			if arr_i != len(arrs) - 1:
				file.write('\t')
		file.write('\n')
	file.close()

def center_waveform(data, end_index, waveform_size, window_size):
	'''
	Centers a captured waveform to have more similar waveform starting points

	parameters: 
		data: 1D array of data points, larger than waveform size
		end_index: index of last point of captured waveform in data
		waveform_size: length of 1D array that can hold entire waveform size
		window_size: length of full window
	'''

	padding = int((window_size-waveform_size)/2)
	# return [data[(end_index-waveform_size)] for i in range(padding)] + data[(end_index-waveform_size):end_index] + [data[end_index]for i in range(padding)]
	return [0 for i in range(padding)] + data[(end_index-waveform_size):end_index] + [0 for i in range(padding)]