import numpy as np
import csv
import glob
import random

motions = input('What motions and amplitudes do you want to examine? ').split(' ')
nums_train = input('How many training files, respectively? ').split(' ')

def data_reformat(motion, num_train):
	file_names = glob.glob('captured_motion_data/4s_'+ motion + '*.txt')
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
		with open(file, 'r') as data:
			reader = csv.reader(data, delimiter='\t')
			out = []
			for row in reader:
				for item in row:
					try:
						out.append(float(item))
					except:
						pass
			return out

	train_data = []
	test_data = []

	for file in train_files:
		train_data.append(get_voltage_vals(file))

	for file in test_files:
		test_data.append(get_voltage_vals(file))

	return (train_data, test_data)

X_train = []
X_test = []

print (list(zip(motions, nums_train)))

for motion, num_train in zip(motions, nums_train):
	(train, test) = data_reformat(motion, int(num_train))
	# print (train, test)
	X_train.append(train)
	X_test.append(test)

def write_file(file_name, arrays_to_write):
	file = open(file_name, 'w')
	for array in arrays_to_write:
		for line in array:
			for item in line:
				file.write(str(item))
				file.write('\t')
			file.write('\n')
	file.close()

write_file('knn_dtw_data/X_train.txt', X_train)
write_file('knn_dtw_data/X_test.txt', X_test)
