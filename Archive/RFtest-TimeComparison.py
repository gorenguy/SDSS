# Import

import numpy as np
import sklearn.ensemble
import time
from RF import *

# Create Real Data
X, x_total, y_total=create2dobs(1500)

# create object IDs that will be just integers
obj_ids = np.arange(len(x_total))

# Create Synthetic Data
X_syn = return_synthetic_data(X)

# Merge Real & Synthetic
X_total, Y_total = merge_work_and_synthetic_samples(X, X_syn)

# declare an RF
N_TRAIN = 100 # number of trees in the forest
rand_f = sklearn.ensemble.RandomForestClassifier(n_estimators=N_TRAIN)
rand_f.fit(X_total, Y_total)

## object's probability
xx, yy = np.meshgrid(np.linspace(0, 140, 100), np.linspace(0, 140, 100))
Z = rand_f.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 0]

Z2=rand_f.predict_proba(X)[:,0]
print('Avg. prob. of Real in RF tree given a Real observation: '),
print(Z2.mean())

# Calculating similarity matrix time comparison
Ntest=20

start=time.time()
for i in xrange(Ntest):
    sim_mat = build_similarity_matrix(rand_f, X)
end=time.time()
print('build_similarity_matrix time:'),
print((end-start)/Ntest)

start=time.time()
for i in xrange(Ntest):
    sim_mat2 = build_similarity_matrix2(rand_f, X)
end=time.time()
print('build_similarity_matrix2 time:'),
print((end-start)/Ntest)

start=time.time()
for i in xrange(Ntest):
    sim_mat3 = build_similarity_matrix3(rand_f, X)
end=time.time()
print('build_similarity_matrix3 time:'),
print((end-start)/Ntest)

print('sim_mat==sim_mat2: '),
print((sim_mat==sim_mat2).all())
print('Similarity matrices size: '),
if (sim_mat.shape == sim_mat3.shape):
    print(sim_mat2.shape)
else: print('Matrices of different sizes!')
print('Number of different values sim_mat,sim_mat3: '),
print((sim_mat!=sim_mat3).sum())
print('Is sim_mat symmetric: '),
print((sim_mat==sim_mat.T).all())
print('Is sim_mat3 symmetric: '),
print((sim_mat3==sim_mat3.T).all())
