import numpy as np
import csv
import glob
import random
from sklearn.externals import joblib 
import os

def get_key(d, desired_motion):
	for l, m in d.items(): 
		# print (m) 
		if m == desired_motion:
			return l

def data_reformat(motion, num_train, labels, trial_name):
	file_names = glob.glob('training_data/' + trial_name + '/4s_'+ motion + '*.txt')
	print (file_names)
	file_names.sort()
	train_files = []
	test_files = []

	train_file_nums = random.sample(range(len(file_names)), num_train)
	test_file_nums = [x for x in range(len(file_names)) if x not in train_file_nums]

	for num in train_file_nums:
		train_files.append(file_names[num])

	for num in test_file_nums:
		test_files.append(file_names[num])

	def get_voltage_vals(file):
		out = [[],[],[],[]]
		for i in range(4):
			with open(file, 'r') as data:
				reader = csv.reader(data, delimiter='\t')
				for row in reader:
					out[i].append(float(row[i]))
		return out

	train_data = []
	test_data = []
	train_labels = []
	test_labels = []

	for file in train_files:
		train_data.append(get_voltage_vals(file))
		train_labels.append([get_key(labels, motion)])

	for file in test_files:
		test_data.append(get_voltage_vals(file))
		test_labels.append([get_key(labels, motion)])

	return (train_data, test_data, train_labels, test_labels)

trial_name = input('Name of trial: ')
if not os.path.exists('/home/pi/Desktop/Code/training_files/' + trial_name):
	os.mkdir('/home/pi/Desktop/Code/training_files/' + trial_name)
motions = input('What motions and amplitudes do you want to examine? ').split(' ')
nums_train = input('How many training files, respectively? ').split(' ')
labels_file = input ('What is the name of the labels file you want to use (include the extension)? Just press enter if you do not have one. ')
labels = joblib.load('labels/' + labels_file) if '.pkl' in labels_file else {0:'A', 1:'E', 2:'I', 3:'O', 4:'U', 5:'purse', 6:'open_mouth', 7:'twitch_small',
				8:'twitch_medium', 9:'smile_small', 10:'smile_medium'}

X_train = []
X_test = []
y_train = []
y_test = []

print (list(zip(motions, nums_train)))

for motion, num_train in zip(motions, nums_train):
	# print (motion, num_train)
	(train_data, test_data, train_labels, test_labels) = data_reformat(motion, int(num_train), labels, trial_name)
	# print (train, test)
	X_train.append(train_data) # contains training data for each motion
	X_test.append(test_data)
	y_train.append(train_labels)
	y_test.append(test_labels)

def write_training_data(file_name, arr):
	file = open(file_name, 'w')
	print ('motions', len(arr))
	for mot in arr:
		print ('runs', len(mot))
		for run in mot:
			if isinstance(run[0], list): #[[elem1][elem2][elem3][elem4]]
				print ('elems', len(run))
				for data_i in range(len(run[0])): #number of datapoints
					for elem_i in range(len(run)): #number of elems
						file.write(str(run[elem_i][data_i]))
						file.write('\t')
					file.write('\n')
				file.write('\n')
			else: #labels
				file.write(str(run[0]))
				file.write('\n')
			# file.write('\n')
	file.close()
# print(y_train)
write_training_data('training_files/' + trial_name + '/X_train.txt', X_train)
write_training_data('training_files/' + trial_name + '/X_test.txt', X_test)
write_training_data('training_files/' + trial_name + '/y_train.txt', y_train)
write_training_data('training_files/' + trial_name + '/y_test.txt', y_test)
