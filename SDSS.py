import pandas as pd
import os
import urllib
import numpy as np
from scipy.interpolate import interp1d
from scipy import signal
from astropy.io import fits
from sklearn.preprocessing import Imputer
import sfdmap
from joblib import Parallel, delayed
import multiprocessing as mp
from functools import partial


def getFitsFiles(gs,path,foldername):

    """
    :param gs: Galaxies pandas dataframe containing the fields 'plate', 'mjd', 'fiberid', run2d'
    :param path: Path to DB folder
    :param foldername: New folder name in DB folder to save data to
    :return: None. Downloads .fits files from dr14 for all galaxies to path/foldername
    """

    print ('*** %s %d %s ****' % ('Getting ', len(gs), ' Galaxies'))
    print('%12s %12s %12s %12s %12s %12s' % ('Count', 'Version', 'Plate', 'mjd', 'Fiber id', 'Status'))

    failed=[]

    newpath = path + foldername + '\\'
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    for i in xrange(len(gs)):
        obj = gs.iloc[i]
        plate=str(obj['plate']).zfill(4)
        mjd = str(obj['mjd'])
        fiberid = str(obj['fiberid']).zfill(4)
        run2d = str(obj['run2d'])

        print('%12s %12s %12s %12s %12s' % (str(i), run2d, plate, mjd, fiberid)),

        try:
            getFitsFile(plate,mjd,fiberid,run2d, newpath)
            print('%12s' % ('Succeeded'))
        except:
            print('%12s' % ('Failed'))
            failed.append(obj.name)

    print('Finished Downloading!'),
    print(' Total failed: ' + str(len(failed)))
    if len(failed)>0:
        print('Failed: ')
        print(gs.ix[failed])


def getFitsFile(plate,mjd,fiberid, run2d,path):

    """
    :param plate: Plate number (string, 4 digits)
    :param mjd: mjd (string)
    :param fiberid: fiber ID (string, 4 digits)
    :param run2d: Version (string)
    :param path: Path for .fits file
    :return: None. Downloads the given .fits file from dr14 to path
    """

    filename = '-'.join(['spec',plate,mjd,fiberid]) + '.fits'
    url = 'https://data.sdss.org/sas/dr14/sdss/spectro/redux/' + run2d + '/spectra/lite/' + plate + '/' + filename
    dest = path + filename

    if not(os.path.isfile(dest)):
        urllib.urlretrieve(url, dest)
    assert os.path.getsize(dest) > 1000, 'Spectra wasn\'t found'


def calcFitsFilenames(gs):
    filenames=[]
    for i in xrange(len(gs)):
        obj = gs.iloc[i]
        plate=str(obj['plate']).zfill(4)
        mjd = str(obj['mjd'])
        fiberid = str(obj['fiberid']).zfill(4)
        filename = '-'.join(['spec',plate,mjd,fiberid]) + '.fits'
        filenames.append(filename)
    return(filenames)


def dataFromFits(filenames, path):
    data=[]
    specobjid=[]
    for i,filename in enumerate(filenames):
        print('Getting Data from .Fits: '),
        hdulist = fits.open(path + filename)
        data_i = hdulist[1].data
        specobjidi=int(np.asscalar(hdulist[2].data['specobjid']))
        hdulist.close()
        data.append(data_i)
        specobjid.append(specobjidi)
        print(str(round(100 * (i + 1) / float(len(filenames)), 3)) + '%')
    return np.array(specobjid), data


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


def deredden_spectrum(wl, spec, E_bv):
    """
    Dereddens a spectrum based on the given extinction_g value and Fitzpatric99 model
    IMPORTANT: the spectrum should be in the observer frame (do not correct for redshift)

    :param wl: loglam
    :param spec: flux
    :param E_bv: m.ebv(ra,dec)
    :return:
    """

    # dust model
    wls = np.array([ 2600,  2700,  4110,  4670,  5470,  6000, 12200, 26500])
    a_l = np.array([ 6.591,  6.265,  4.315,  3.806,  3.055,  2.688,  0.829,  0.265])
    f_interp = interp1d(wls, a_l, kind="cubic")

    a_l_all = f_interp(wl)
    #E_bv = extinction_g / 3.793
    A_lambda = E_bv * a_l_all
    spec_real = spec * 10 ** (A_lambda / 2.5)

    return spec_real


def remove_bad_pixels(spec, ivar):
    """
    Puts to Nan pixels with zero inverse variance
    :param spec: Spectrum
    :param ivar:
    :return: Spectrum with Nan for bad pixels
    """
    spec[ivar == 0] = np.nan

    return spec


def zero_to_nan(arr):
    """
    Replace zeros with nans

    Note: doing this because there are empty entries in the matrix which are set to zero (Due to different objects having different number of pixels)
    :param arr: spec
    :return: spec zero to nan
    """

    arr[arr == 0] = np.nan
    return arr


def de_redshift(wave, z):
    """
    Switch to rest frame wave length

    :param wave: wl in observer frame
    :param z: Red shift
    :return: wl in rest frame
    """
    wave = wave / (1 + z)
    return wave


def calcGrid(gs,data):

    # Calculate a grid its length (number of wavelengths available) is lower than 95% of the available wavelengths of the observations

    wlLen = np.array([len(x['loglam']) for x in data])
    WlLenVal = np.percentile(wlLen, 5)
    gridIndex = np.argmin(np.abs(WlLenVal - wlLen))
    grid = de_redshift(10 ** data[gridIndex]['loglam'], gs.iloc[gridIndex]['z'])

    return(grid)


def same_grid_single(wave_common, wave_orig, spec_orig):
    """
    Putting a single spectrum on the common wavelength grid

    :param wave_common: wl vector
    :param wave_orig:
    :param spec_orig:
    :return:
    """
    spec = np.interp(wave_common, wave_orig, spec_orig, left=np.nan, right=np.nan)

    return spec


def medfilt(spec, size=5):
    return signal.medfilt(spec, size)


def norm_spectrum(spec):
    """
    Normalize spectrum - divide by median (clipped to one)

    :param spec: Spectrum
    :return: Normalized Spectrum
    """
    spec_norm = np.nanmedian(spec)
    if spec_norm >= 1:
        spec = (spec / spec_norm)
    else:
        spec = spec + (1 - spec_norm)

    return spec


def calcSpecDF(filenames,grid,gs,data,specobjid):
    specDF = np.zeros((len(filenames), len(grid)))
    ebvMap = sfdmap.SFDMap('E:\DB\sfddata-master')

    for i in xrange(len(filenames)):
        print('Calculating Spectrum: '),
        gi = gs.iloc[i]
        wli = 10 ** data[i]['loglam']
        fluxi = data[i]['flux']
        ivari = data[i]['ivar']
        assert specobjid[i] == gi['specobjid'], 'Files do not match galaxies dataframe'
        ra = gi['ra']
        dec = gi['dec']
        ebvi = ebvMap.ebv(ra, dec)
        speci = deredden_spectrum(wli, fluxi, ebvi)
        speci = remove_bad_pixels(speci, ivari)
        speci = zero_to_nan(speci)
        wli = de_redshift(wli, gi['z'])
        speci = same_grid_single(grid, wli, speci)
        speci = medfilt(speci)
        speci = norm_spectrum(speci)
        specDF[i] = speci
        print(str(round(100 * (i+1) / float(len(filenames)),3)) + '%')
    return specDF

