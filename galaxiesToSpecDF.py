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

# Pickles path
picklesPath = r'E:\DB\pickles\\'
if not os.path.exists(picklesPath):
    os.makedirs(picklesPath)

# DB path
path = r'E:\DB\\'
filename = 'best20kSNR'
filetype = 'csv'

# Load galaxies data
# gs=pd.read_csv(path+filename+'.'+filetype, header = 1)
# gs.to_pickle(picklesPath + 'gs.pkl')
gs = pd.read_pickle(picklesPath + 'gs.pkl')

# Get fits files
# getFitsFiles(gs,path,filename)

# Shrink galaxies data
gs = gs[:2000]

# Calculate observations filenames
filenames=calcFitsFilenames(gs)
path += filename +'\\'

# Get data from .fits files
# specobjid, data = dataFromFits(filenames,path)
# np.save(picklesPath + 'data.npy', data)
# np.save(picklesPath + 'specobjid.npy', specobjid)
data = np.load(picklesPath + 'data.npy')
specobjid = np.load(picklesPath + 'specobjid.npy')

# Calculate grid of wavelength
# grid = calcGrid(gs,data)
# np.save(picklesPath + 'grid.npy',grid)
grid = np.load(picklesPath + 'grid.npy')

# Calculate spectra DF
specDF = calcSpecDF(filenames,grid,gs,data,specobjid)

nanRemovalThreshold = 0.1
properNanCountMask = np.isnan(specDF).sum(1) < nanRemovalThreshold * specDF.shape[1]
print('%s %d/%d %s' % ('Nan Values Counts: Removing', np.bitwise_not(properNanCountMask).sum(),len(np.bitwise_not(properNanCountMask)), 'observations'))
specDF = specDF[properNanCountMask]
specDF = Imputer('NaN', 'median').fit_transform(specDF)
specobjid = specobjid[properNanCountMask]
gs = gs[properNanCountMask]
gs.index=range(len(gs))
data = data[properNanCountMask]
np.save(picklesPath + 'grid_final.npy',grid)
np.save(picklesPath + 'specDF_final.npy',specDF)
np.save(picklesPath + 'specobjid_final.npy', specobjid)
np.save(picklesPath + 'data_final.npy', data)
gs.to_pickle(picklesPath + 'gs_final.pkl')