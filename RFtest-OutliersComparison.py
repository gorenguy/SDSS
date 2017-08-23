# Import

import numpy as np
import sklearn
import sklearn.ensemble
import matplotlib.pyplot as plt
import scipy.spatial
import time
from RF import *
from useful import *
from SDSS import *

# Create Real Data
X, x_total, y_total=create2dobs(3000)

# create object IDs that will be just integers
obj_ids = np.arange(len(x_total))

# Create Synthetic Data
X_syn = return_synthetic_data(X)

# Merge Real & Synthetic
X_total, Y_total = merge_work_and_synthetic_samples(X, X_syn)

# declare an RF
N_TRAIN = 200 # number of trees in the forest
rand_f = sklearn.ensemble.RandomForestClassifier(n_estimators=N_TRAIN)
rand_f.fit(X_total, Y_total)

## Object's probability
Z2=rand_f.predict_proba(X)[:,0]
print('Avg. prob. of Real in RF tree given a Real observation: '),
print(Z2.mean())

# Calculating similarity matrix time comparison
start=time.time()
dis_mat = calcDisMat(rand_f, X)
end=time.time()
print('calcDisMat time:'),
print(end-start)

# Calculate Wall By RF
wall = calculateWall(dis_mat)

#plotWeirdnessHist(wall)

# Calculate Wnum by RF
num=5
wnum = calculateWnum(dis_mat, obj_ids, num)

#plotWeirdnessHist(wnum)

# Calculate Wall by euclidean distance
dis_matEuc=scipy.spatial.distance_matrix(X,X)
dis_matEuc = dis_matEuc / np.amax(dis_matEuc)

wallEuc=calculateWall(dis_matEuc)

plotWeirdnessHist(wallEuc)

# Calculate Wnum by euclidean distance
wnumEuc=calculateWnum(dis_matEuc,obj_ids,num)

#plotWeirdnessHist(wnumEuc)

N_outliers = 50

wnum_outliers = np.sort(wnum)[::-1][:N_outliers]
obj_ids_outliers = obj_ids[np.argsort(wnum)][::-1][:N_outliers]

wnum_outliersEuc = np.sort(wnumEuc)[::-1][:N_outliers]
obj_ids_outliersEuc = obj_ids[np.argsort(wnumEuc)][::-1][:N_outliers]

plt.rcParams['figure.figsize'] = 5, 5
plt.title("Data and outliers")
plt.plot(X[:,0], X[:,1], "ok", label="input data", markersize=4)
plt.plot(X[obj_ids_outliers, 0], X[obj_ids_outliers, 1], "om", label="RF outliers", markersize=8)
plt.plot(X[obj_ids_outliersEuc, 0], X[obj_ids_outliersEuc, 1], "*g", label="Euclidean outliers", markersize=8)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend(loc="best")
