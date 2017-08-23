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

def dataFromFits_par(filenames, path):

    """ Parallel Version """

    data=[]
    specobjid=[]

    pool = mp.Pool()
    f = partial(dataFromFit_i, path,filenames)
    res = pool.map(f, range(len(filenames)))

    data = []
    specobjid = []
    for tup in sorted(res, key=lambda x: x[0]):
        data.append(tup[1])
        specobjid.append(tup[2])

    return np.array(specobjid), data


def dataFromFit_i(path,filenames, i):
    hdulist = fits.open(path + filenames[i])
    data_i = hdulist[1].data
    specobjidi = int(np.asscalar(hdulist[2].data['specobjid']))
    hdulist.close()
    return (i, data_i, specobjidi)


if __name__=='__main__':

    path = r'E:\DB\\'
    filename = 'best20kSNR'

    picklesPath = r'E:\DB\pickles\\'
    if not os.path.exists(picklesPath):
        os.makedirs(picklesPath)

    gs = pd.read_pickle(picklesPath + 'gs.pkl')

    gs = gs[:3000]

    filenames = calcFitsFilenames(gs)

    path += filename + '\\'

    print('Getting data from fits (parallel): '),
    start=time.time()
    specobjid_par, data_par = dataFromFits_par(filenames, path)
    end=time.time()
    print(end - start)

    print('Getting data from fits: '),
    start = time.time()
    specobjid, data = dataFromFits(filenames, path)
    end = time.time()
    print(end - start)

    print((specobjid == specobjid_par).all())

    flag = True
    for i in xrange(len(data)):
        flag = flag & (data[i] == data_par[i]).all()
    print(flag)