# based on the code provided on github by markdregan on github
# https://github.com/markdregan/K-Nearest-Neighbors-with-Dynamic-Time-Warping
# edited by dougb and faritatasnim for use with Conformable Decoders group AlN sensor
# original dtw algorithm deleted and replaced with slaypni's fastdtw algorithm based on the paper:
# https://cs.fit.edu/~pkc/papers/tdm04.pdf

import sys
import collections
import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
from scipy.spatial.distance import squareform
from sklearn.externals import joblib 
import scipy.spatial.distance as distance
import fastdtw

plt.style.use('bmh')

try:
    from IPython.display import clear_output
    have_ipython = True
except ImportError:
    have_ipython = False

class KnnDtw(object):

    def __init__(self, n_neighbors = 10, radius = 1, subsample_step = 1, dist = None):
        self.n_neighbors = n_neighbors
        self.radius = radius
        self.subsample_step = subsample_step
        self.dist = dist
    
    def fit(self, x, l):
        
        self.x = x
        self.l = l

    def cos_sim(p):
        return lambda a, b: abs(a - b)/(np.linalg.norm(np.atleast_1d(a) - np.atleast_1d(b), 1))
    
    def _dist_matrix(self, x, y):
        
        # Compute the distance matrix        
        dm_count = 0
        
        x_s = np.shape(x)
        y_s = np.shape(y)
        print (x_s, y_s)
        dm = np.zeros((x_s[0], y_s[0])) 
        dm_size = x_s[0]*y_s[0]*4

        p = ProgressBar(dm_size)

        for i in range(0, x_s[0]):
            for j in range(0, y_s[0]):
                dm[i,j] = 0
                for k in range (0, y_s[1]):
                    if k != 1 and k!= 2:
                        print (x[i,k, ::self.subsample_step].shape, 
                            y[j,k, ::self.subsample_step].shape)
                        to_add = (fastdtw.fastdtw(x[i,k, ::self.subsample_step], 
                            y[j,k, ::self.subsample_step], radius = self.radius)[0])**2
                        print ('dm component', to_add)
                        dm[i,j] += to_add

                    # Update progress bar
                    dm_count += 1
                    p.animate(dm_count)

        for i in range(0, x_s[0]):
            for j in range(0, y_s[0]):
                dm[i,j] = np.sqrt(dm[i,j])
        print (dm)
        return dm
    
    def repeats(self,item,arr):
        inds = []
        for ind,i in enumerate(arr):
            if i == item:
                inds.append(ind)
        return inds

    def score_repeats(self, repeats):
        score = 0
        for item in repeats:
            score += (self.n_neighbors-item)
        return score

    def find_weighted_modes(self, data_labels):
        weighted_modes = []
        for labels in data_labels:
            weighted_modes.append([])
            repeats = []
            repeats_score = []
            for item in list(set(labels)):
                repeats.append((item, self.repeats(item, labels)))
                repeats_score.append((item, self.score_repeats(self.repeats(item, labels))))
            repeats_score.sort(key=lambda l:l[1], reverse = True)
            # print (repeats)
            # print (repeats_score)
            weighted_modes[-1].append(repeats_score)
        # print ('weighted modes', weighted_modes)
        return weighted_modes

    def predict(self, x):
        dm = self._dist_matrix(x, self.x)

        # Identify the k nearest neighbors
        knn_idx = dm.argsort()[:, :self.n_neighbors]

        # Identify k nearest labels
        knn_labels = self.l[knn_idx]
        print (knn_labels)

        weighted_values = self.find_weighted_modes(knn_labels)
        weighted_modes = [i[0] for i in weighted_values]
        weighted_probs = [[(j[l][0], j[l][1]/(.5*self.n_neighbors*(self.n_neighbors+1))) for l,val in enumerate(j)] for j in weighted_modes]
        print ('weighted modes', weighted_modes)
        # print ('weighted probs', weighted_probs)

        # print ('num neighbors', self.n_neighbors)
        # print ('radius', self.radius)
        
        # Model Label
        # mode_data = mode(knn_labels, axis=1)
        # mode_label = mode_data[0]
        mode_label = np.array([np.array([i[0][0]]) for i in weighted_probs])
        # print (mode_label)
        mode_proba = np.array([np.array([i[0][1]]) for i in weighted_probs])
        # mode_proba = mode_data[1]/self.n_neighbors
        # print (mode_proba)

        return mode_label.ravel(), mode_proba.ravel(), weighted_probs

class ProgressBar:
    """This progress bar was taken from PYMC
    """
    def __init__(self, iterations):
        self.iterations = iterations
        self.prog_bar = '[]'
        self.fill_char = '*'
        self.width = 40
        self.__update_amount(0)
        if have_ipython:
            self.animate = self.animate_ipython
        else:
            self.animate = self.animate_noipython

    def animate_ipython(self, iter):
        print ('\r', self,)
        sys.stdout.flush()
        self.update_iteration(iter + 1)

    def update_iteration(self, elapsed_iter):
        self.__update_amount((elapsed_iter / float(self.iterations)) * 100.0)
        self.prog_bar += '  %d of %s complete' % (elapsed_iter, self.iterations)

    def __update_amount(self, new_amount):
        percent_done = int(round((new_amount / 100.0) * 100.0))
        all_full = self.width - 2
        num_hashes = int(round((percent_done / 100.0) * all_full))
        self.prog_bar = '[' + self.fill_char * num_hashes + ' ' * (all_full - num_hashes) + ']'
        pct_place = (len(self.prog_bar) // 2) - len(str(percent_done))
        pct_string = '%d%%' % percent_done
        self.prog_bar = self.prog_bar[0:pct_place] + \
            (pct_string + self.prog_bar[pct_place + len(pct_string):])

    def __str__(self):
        return str(self.prog_bar)

# Import the facial micromotion dataset  
# These files were produced from the vectors in the twitching and smiling spreadsheet tabs.
# The first 4 instances were used to create X_train.txt files, the last to create X_test.txt

'''
x_train_file = open('knn_dtw_data/X_train_4.txt', 'r')
y_train_file = open('knn_dtw_data/y_train.txt', 'r')

x_test_file = open('knn_dtw_data/X_test_4.txt', 'r')
y_test_file = open('knn_dtw_data/y_test.txt', 'r')

# Create empty lists
x_train = []
y_train = []
x_test = []
y_test = []

# Mapping table for classes
labels = {0:'A', 1:'E', 2:'I', 3:'O', 4:'U', 5:'purse', 6:'open_mouth', 7:'twitch_small',
            8:'twitch_medium', 9:'smile_small', 10:'smile_medium'}

# Loop through datasets
for x in x_train_file:
    x_train.append([float(ts) for ts in x.split()[:4000]])
    
for y in y_train_file:
    y_train.append(int(y.rstrip('\n')))
    
for x in x_test_file:
    x_test.append([float(ts) for ts in x.split()[:4000]])
    
for y in y_test_file:
    y_test.append(int(y.rstrip('\n')))
    
# Convert to numpy for efficiency
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
print(x_test.shape, x_train.shape)
y_test = np.array(y_test)

nn = 1
ww = 1
m = KnnDtw(n_neighbors=nn, radius=ww, subsample_step=1)
m.fit(x_train, y_train)
joblib.dump(m, 'knn_model_nn' + str(nn) + '_ww_' + str(ww) + '.pkl')

knn_model = joblib.load('knn_model_nn' + str(nn) + '_ww_' + str(ww) + '.pkl')
label, proba = knn_model.predict(np.array(x_test))

print (label, proba)

from sklearn.metrics import classification_report, confusion_matrix

print ("nn = ", nn, " ww = ", ww)
print (classification_report(label, y_test,
                            target_names=[l for l in labels.values()]))

conf_mat = confusion_matrix(label, y_test)

fig = plt.figure(figsize=(11,11))
width = np.shape(conf_mat)[1]
height = np.shape(conf_mat)[0]

res = plt.imshow(np.array(conf_mat), cmap=plt.cm.viridis, interpolation='nearest')
for i, row in enumerate(conf_mat):
    for j, c in enumerate(row):
        if c>0:
            plt.text(j-.2, i+.1, c, fontsize=16)
            
cb = fig.colorbar(res)
plt.title('Confusion Matrix')
_ = plt.xticks(range(11), [l for l in labels.values()], rotation=90)
_ = plt.yticks(range(11), [l for l in labels.values()])

plt.show()
'''