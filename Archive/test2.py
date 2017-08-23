if __name__ == '__main__':

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
    from functools import partial
    import time

    import multiprocessing as mp
    import random

    picklesPath = r'E:\DB\pickles\\'
    if not os.path.exists(picklesPath):
        os.makedirs(picklesPath)

    specDF = np.load(picklesPath + 'specDF_final.npy')
    specDF=np.vstack((specDF,specDF))
    rand_f = np.load(picklesPath + 'rand_f.pkl')

    start=time.time()
    print('Building RF distance matrix (parallel)... '),
    dis_matPar = calcDisMat_par(rand_f, specDF)
    end=time.time()
    print ('Done! (time: ' + str(round(end-start,3)) + ')')

    start=time.time()
    print('Building RF distance matrix... '),
    dis_mat = calcDisMat(rand_f, specDF)
    end=time.time()
    print ('Done! (time: ' + str(round(end-start,3)) + ')')

    print((dis_mat == dis_matPar).all())