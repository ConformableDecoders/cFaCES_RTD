import sys
import os
import collections
import itertools
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import mode
from scipy.spatial.distance import squareform
from sklearn.externals import joblib 
from sklearn.metrics import classification_report, confusion_matrix
from knn_dtw_training import KnnDtw
from datetime import datetime as dt

plt.style.use('bmh')
# train classifier
def train_classifier(nn, ww, labels, trial_name):
	x_train_file = open('training_files/' + trial_name + '/X_train.txt', 'r')
	y_train_file = open('training_files/' + trial_name + '/y_train.txt', 'r')

	x_test_file = open('training_files/' + trial_name + '/X_test.txt', 'r')
	y_test_file = open('training_files/' + trial_name + '/y_test.txt', 'r')

	if not os.path.exists('/home/pi/Desktop/Code/saved_knn_models/' + trial_name):
		os.mkdir('/home/pi/Desktop/Code/saved_knn_models/' + trial_name)

	if not os.path.exists('/home/pi/Desktop/Code/accuracy/' + trial_name):
		os.mkdir('/home/pi/Desktop/Code/accuracy/' + trial_name)

	# Loop through datasets
	def reshape_data(file, wv_len):
		y = [x for x in file]
		z = [x.split() for x in y]
		a = [z[wv_len*(i):wv_len*(i+1)-1] for i in range(int(len(z)/wv_len))]
		b = np.array([np.array(x) for x in a])
		c = np.reshape(b, (b.shape[0], b.shape[2], b.shape[1]))
		#c[i][j] references element j's waveform for run i
		#motions ID of run is indicated by corresponding label in y_train or y_test file
		return c
	    
	x_train = reshape_data(x_train_file, 401)
	y_train = np.array([int(y) for y in y_train_file])
	x_test = reshape_data(x_test_file, 401)    
	y_test = np.array([int(y) for y in y_test_file])
	    
	# Convert to numpy for efficiency
	print(x_test.shape, x_train.shape)

	nn = nn
	ww = ww
	m = KnnDtw(n_neighbors=nn, radius=ww, subsample_step=1)
	m.fit(x_train, y_train)
	label, proba, weighted_probs = m.predict(np.array(x_test))
	report_metrics(nn, ww, labels, label, proba, x_test, y_test, trial_name)
	joblib.dump(m, 'saved_knn_models/' + trial_name + '/knn_model_nn' + str(nn) + '_ww_' + str(ww) + '.pkl')

# test data
def predict_classification(waveform, nn, ww, trial_name):
	# input waveform should be an np array
	knn_model = joblib.load('saved_knn_models/' + trial_name + '/knn_model_nn' + str(nn) + '_ww_' + str(ww) + '.pkl')
	label, proba, weighted_probs = knn_model.predict(np.array(waveform))
	return label, proba, weighted_probs


# report metrics
def report_metrics(nn, ww, labels, label, proba, x_test, y_test, trial_name):
	print ('labels in report metrics', labels)

	print ("nn = ", nn, " ww = ", ww)
	print (classification_report(label, y_test,
	                            target_names=[l for l in labels.values()]))
	num_labels = len(labels.values())

	conf_mat = confusion_matrix(label, y_test)

	fig = plt.figure(figsize=(num_labels*2,num_labels*2))
	width = np.shape(conf_mat)[1]
	height = np.shape(conf_mat)[0]

	res = plt.imshow(np.array(conf_mat), cmap=plt.cm.viridis, interpolation='nearest')
	for i, row in enumerate(conf_mat):
	    for j, c in enumerate(row):
	        if c>0:
	            plt.text(j-.2, i+.1, c, fontsize=16)
	            
	cb = fig.colorbar(res)
	plt.title('Confusion Matrix')
	_ = plt.xticks(range(num_labels), [l for l in labels.values()], rotation=90)
	_ = plt.yticks(range(num_labels), [l for l in labels.values()])

	figname = input('Descriptor of accuracy file: ')
	savename = 'accuracy/' + trial_name + '/testing_accuracy_' + figname + '_' + dt.now().strftime('%d_%m_%y_%H_%M') + '.png'

	plt.savefig(savename, dpi=300)
	plt.show()

