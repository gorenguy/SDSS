from SDSS import *
from RF import *
import pandas as pd
import os
import urllib
from astropy.io import fits
from scipy import interpolate
import matplotlib.pyplot as plt
import numpy as np
import statistics
import pickle
import scipy.spatial
import sklearn
import sklearn.ensemble
import time
from useful import *

picklesPath = r'E:\DB\pickles\\'
if not os.path.exists(picklesPath):
    os.makedirs(picklesPath)

# Load data
print('Loading data...'),
start=time.time()
grid = np.load(picklesPath + 'grid_final.npy')
specDF = np.load(picklesPath + 'specDF_final.npy')
gs = pd.read_pickle(picklesPath + 'gs_final.pkl')
specobjid = np.load(picklesPath + 'specobjid_final.npy')
data = np.load(picklesPath + 'data_final.npy')
obj_ids = np.array(range(len(specDF)))
assert (gs.specobjid == specobjid).all(), 'Data mismatch'
end=time.time()
print ('Done! (time: ' + str(round(end-start,3)) + ')')

# Create Synthetic data
specDF_syn = return_synthetic_data(specDF)
specDF_merged, specDF_merged_classes = merge_work_and_synthetic_samples(specDF, specDF_syn)

# Build forest
nTrain = 200
rand_f = sklearn.ensemble.RandomForestClassifier(n_estimators=nTrain)
start=time.time()
print('Build forest... '),
rand_f.fit(specDF_merged, specDF_merged_classes)
end=time.time()
print ('Done! (time: ' + str(round(end-start,3)) + ')')

# Real classification probability
realProb = rand_f.predict_proba(specDF)[:,0]
print('Avg. prob. of Real in RF tree given a Real observation: '),
print(realProb.mean())

# Build RF distance matrix
start=time.time()
print('Building RF distance matrix... '),
dis_mat = calcDisMat(rand_f, specDF)
end=time.time()
print ('Done! (time: ' + str(round(end-start,3)) + ')')

# Calculate Wnum
num = 5
wnum = calculateWnum(dis_mat, obj_ids, num)

"""
# Build euclidean distance matrix
start=time.time()
print('Building  euclidean distance matrix... '),
dis_matEuc = scipy.spatial.distance_matrix(specDF,specDF)
end=time.time()
print ('Done! (time: ' + str(round(end-start,3)) + ')')
wnumEuc = calculateWnum(dis_matEuc, obj_ids, num)
"""

# plt.hist(wnumEuc, bins=50)
plt.hist(wnum, bins=50)

# Calculate outliers
N_outliers = 50

wnum_outliers = np.sort(wnum)[::-1][:N_outliers]
obj_ids_outliers = obj_ids[np.argsort(wnum)][::-1][:N_outliers]

# wnum_outliersEuc = np.sort(wnumEuc)[::-1][:N_outliers]
# obj_ids_outliersEuc = obj_ids[np.argsort(wnumEuc)][::-1][:N_outliers]