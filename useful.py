import pickle
import matplotlib.pyplot as plt


def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    with open(filename, 'rb') as input:
        return(pickle.load(input))


def plotGalaxySpec(specDF, grid, index):
    plt.plot(grid,specDF[index])
    plt.title('Galaxy Spectrum ' + str(index))
    plt.xlabel('Wavelength [A]')
    plt.ylabel('Normalized Intensity')


def plotWeirdnessHist(w):

    """
    :param w: Weirdness score vector
    :return: None. Plots weirdness histogram
    """

    plt.rcParams['figure.figsize'] = 6, 4
    plt.title("Weirdness Score Histogram")
    plt.hist(w, bins=60, color="g")
    plt.ylabel("N")
    plt.xlabel("weirdness score")