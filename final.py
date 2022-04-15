# imports
from sklearn.linear_model import LinearRegression
from scipy.signal import find_peaks
from os.path import isfile, join
import matplotlib.pyplot as plt
from os import listdir
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import math


# linear regression
def linear(m, x, b):
    return m * x + b


# Define the Gaussian function
def gaussian(x, amp1, cen1, sigma1):
    return amp1 * (1 / (sigma1 * (np.sqrt(2 * np.pi)))) * (np.exp((-1.0 / 2.0) * (((x - cen1) / sigma1) ** 2)))


# define function for exponential decay regression
def exponential(x, i_0, u):
    return i_0 * (math.e ** (-u * x))


# convert axis
def convert_axis(x):
    return (x * 1.711) + -16.952


# function to get efficient of a given energy level
def efficiency_of_detector(x):
    return (-0.0008952857335309725 * x) + 2.418254345530344


# function to convert counts to grams
def convert_counts_to_grams(x):
    sec_per_hl = 1.289 * 365 * 86400  # years * days * seconds
    l = .693 / sec_per_hl  # number of half lives
    decays = x / l  # number of decays expected
    mols = decays / 6.022e23  # convert k's to mols
    grams = mols * 39.9639982  # mass of k-40
    return (grams / 0.000117)[0]  # only ~0.0117% of k is k-40


# table data
calibration_energies = np.array([122.0, 1173.2, 1332.5, 661.7, 834.8, 511.0, 1274.5])
collection_times = np.array([300, 300, 300, 120, 300, 300, 300])
yields = np.array([0.86, 1, 1, 0.85, 1, 1.8, 1])
activity = np.array([1.19, 0.84, 0.84, 1.12, 1.07, 1.16, 1.16]) * (10 ** -6) * (3.7 * (10 ** 10))
# ---------------------------------------------- PART 2 -------------------------------------------------------

# directory for the files
DIR = 'data/lab_7/calibration'

# read available model files=
files = [f for f in listdir(DIR) if isfile(join(DIR, f)) and not f.startswith('.')]
files = sorted(files)

peak_indices = []
decays = []
for i, cali_trial in enumerate(files):
    trial = pd.read_csv(DIR + '/' + cali_trial)

    X, Y = np.linspace(0, len(trial) - 1, len(trial)), trial[trial.columns[0]]

    # find peaks and peak differences
    peaks, _ = find_peaks(Y, distance=50, height=max(Y) * 0.5, prominence=0.8)

    # edit found peaks to be the real ones since it is too hard to describe in code
    if i == 2:
        peaks = np.delete(peaks, 0)
    if i == 3:
        peaks = np.delete(peaks, 0)
    if i == 4:
        peaks = np.append(peaks, 755)

    # record the peaks indices
    for peak in peaks:
        peak_indices.append(peak)
        decays.append(Y[peak])

    # add the data to the figure
    plt.vlines(X[peaks], ymin=0, ymax=max(Y), linestyles='dotted', label='detected peak')
    plt.plot(trial, label=cali_trial)

    # add labels and axis to the figure
    plt.title("Gamma Spectroscopy Calibration Trials")
    plt.xlabel('Channel Number (#)')
    plt.ylabel('Count (#)')
    plt.legend()
    plt.figure()

# run a regression on the indexes and the peak energies and show the plot
popt_linear, pcov_linear = curve_fit(linear, peak_indices, calibration_energies)
perr = np.sqrt(np.diag(pcov_linear))
plt.scatter(calibration_energies, peak_indices, c='r', label='data')
x_hat = np.linspace(min(peak_indices), max(peak_indices), 100)
y_hat = linear(popt_linear[0], x_hat, popt_linear[1])
plt.plot(y_hat, x_hat, c='k', label='line of best fit')
plt.xlabel('Channel Number (#)')
plt.ylabel('Peak Energy (keV)')
plt.title('Regression For Peak Energy vs. Channel Number')
plt.legend()
print('Energy = {0} (Channel #) + {1}'.format(round(popt_linear[0], 3), round(popt_linear[1], 3)))
print('error in the slope and intercept are', perr)
plt.show()

# ---------------------------------------------- PART 3 -------------------------------------------------------

# directory for the files
DIR = 'data/lab_7/gaussians'

# read available model files=
files = [f for f in listdir(DIR) if isfile(join(DIR, f)) and not f.startswith('.')]
files = sorted(files)

expected_decays = collection_times * yields * activity

decays = []
for i, cali_trial in enumerate(files):
    trial = pd.read_csv(DIR + '/' + cali_trial)

    X, Y = np.linspace(0, len(trial) - 1, len(trial)), trial[trial.columns[0]]
    Y = Y - min(Y)
    decays.append(np.sum(Y))

    # fit a gaussian curve
    popt_gauss, pcov_gauss = curve_fit(gaussian, X, Y, p0=[1, 9, 1])
    amp, mu, std = popt_gauss

    # add the data to the figure
    plt.scatter(X, Y, label=cali_trial, c='k')
    plt.plot(X, gaussian(x=X, amp1=amp, cen1=mu, sigma1=std), label='fit', c='r')

    # add labels and axis to the figure
    plt.title("Gamma Spectroscopy Calibration Trials")
    plt.xlabel('Channel Number (#)')
    plt.ylabel('Count (#)')
    plt.legend()
    plt.figure()

efficiency = (np.array(decays) / expected_decays) * 100

# run a regression on the indexes and the peak energies and show the plot
popt_linear, pcov_linear = curve_fit(linear, calibration_energies, efficiency)
perr = np.sqrt(np.diag(pcov_linear))
plt.scatter(calibration_energies, efficiency, c='r', label='data')
x_hat = np.linspace(min(calibration_energies), max(calibration_energies), 100)
y_hat = linear(popt_linear[0], x_hat, popt_linear[1])
plt.plot(x_hat, y_hat, c='k', label='line of best fit')
plt.xlabel('Energy Value (keV)')
plt.ylabel('Efficiency (%)')
plt.title('Regression For Efficiency vs. Energy Value')
plt.legend()
print('Efficiency = {0} (Energy) + {1}'.format(popt_linear[0], popt_linear[1]))
print('error in the slope and intercept are', perr)
plt.show()

# ---------------------------------------------- PART 4 -------------------------------------------------------

# directory for the files
DIR = 'data/lab_7/unknown_isotope/unk.dat'


trial = pd.read_csv(DIR)

X, Y = np.linspace(0, len(trial) - 1, len(trial)), trial[trial.columns[0]]
X = convert_axis(X)

# find peaks and peak differences
peaks, _ = find_peaks(Y, height=np.mean(Y) * 1.5, distance=40)
print('For Unknown Element, Gamma Wavelength Peaks Are:', X[peaks], 'keV')

# link to see table of elements and corresponding energy peaks
# https://www.cpp.edu/~pbsiegel/bio431/genergies.html

# add the data to the figure
plt.vlines(X[peaks], ymin=0, ymax=max(Y), linestyles='dotted', label='detected peak')
plt.plot(X, Y, label='unknown isotope data')
plt.legend()
plt.title("Gamma Spectroscopy Unknown Isotope Trial")
plt.ylabel('Count (#)')
plt.xlabel('Energy (keV)')
plt.show()

# ---------------------------------------------- PART 5 -------------------------------------------------------

# directory for the files
DIR = 'data/lab_7/attenuation/Al'

# read available model files=
files = [f for f in listdir(DIR) if isfile(join(DIR, f)) and not f.startswith('.')]
files = sorted(files)
print(files)

dist = []
xray_peaks = []
for file in files:
    trial = pd.read_csv(DIR + '/' + file)

    # determine the distance
    dist.append(float(file[2:-4]))

    X, Y = np.linspace(0, len(trial) - 1, len(trial)), trial[trial.columns[0]]
    X = convert_axis(X)
    peaks, _ = find_peaks(Y, height=np.mean(Y) * 1.5, width=10)
    area_under_curve = X[peaks[0] - 15: peaks[0] + 15]
    xray_peaks.append(sum(area_under_curve))

    # plot the data
    plt.vlines(X[peaks][0], ymin=0, ymax=max(Y), linestyles='dotted', label='detected peak')
    plt.fill_between(X[peaks[0] - 25: peaks[0] + 25], 0, max(Y), color='green', alpha=0.25, lw=0, label='area under peak')
    plt.plot(X, Y, label=file)

    # add labels and axis to the figure
    plt.title("Gamma Spectroscopy Calibration Trials (Al)")
    plt.xlabel('Energy (keV)')
    plt.ylabel('Count (#)')
    plt.legend()
    plt.figure()


# fit an exponential curve
popt_exp, pcov_exp = curve_fit(exponential, xdata=dist, ydata=xray_peaks)
i_0, u = popt_exp
perr = np.sqrt(np.diag(pcov_exp))

plt.scatter(dist, xray_peaks, c='r', label='data')
x_hat = np.linspace(0, 3, 100)
y_hat = exponential(x_hat, i_0, u)
plt.plot(x_hat, y_hat, c='k', label='line of best fit')
plt.xlabel('Thickness')
plt.ylabel('Gamma Intensity')
plt.title('Gamma Intensity vs. Thickness For Al')
plt.legend()
print('Al Attenuation Coefficient = {0}'.format(u))
print('Al I0 Coefficient = {0}'.format(i_0))
print('error in I0, u are:', perr)

plt.show()

# directory for the files
DIR = 'data/lab_7/attenuation/Pb'

# read available model files
files = [f for f in listdir(DIR) if isfile(join(DIR, f)) and not f.startswith('.')]
files = sorted(files)

dist = []
xray_peaks = []
for file in files:
    trial = pd.read_csv(DIR + '/' + file)

    # determine the distance
    dist.append(float(file[2:-4]))

    X, Y = np.linspace(0, len(trial) - 1, len(trial)), trial[trial.columns[0]]
    X = convert_axis(X)
    peaks, _ = find_peaks(Y, height=np.mean(Y) * 1.5, width=10)
    area_under_curve = X[peaks[-1] - 25: peaks[-1] + 25]
    xray_peaks.append(sum(area_under_curve))

    # plot the data
    plt.vlines(X[peaks][-1], ymin=0, ymax=max(Y), linestyles='dotted', label='detected peak')
    plt.fill_between(X[peaks[-1] - 25: peaks[-1] + 25], 0, max(Y), color='green', alpha=0.25, lw=0, label='area under peak')
    plt.plot(X, Y, label=file)

    # add labels and axis to the figure
    plt.title("Gamma Spectroscopy Calibration Trials (Pb)")
    plt.xlabel('Energy (keV)')
    plt.ylabel('Count (#)')
    plt.legend()
    plt.figure()

plt.show()

# fit an exponential curve
popt_exp, pcov_exp = curve_fit(exponential, dist, xray_peaks)
i_0, u = popt_exp
perr = np.sqrt(np.diag(pcov_exp))


plt.scatter(dist, xray_peaks, c='r', label='data')
x_hat = np.linspace(0, 15, 100)
y_hat = exponential(x_hat, i_0, u)
plt.plot(x_hat, y_hat, c='k', label='line of best fit')
plt.xlabel('Thickness')
plt.ylabel('Gamma Intensity')
plt.title('Gamma Intensity vs. Thickness For Pb')
plt.legend()
print('Pb Attenuation Coefficient = {0}'.format(u))
print('Pb I0 Coefficient = {0}'.format(i_0))
print('error in I0, u are:', perr)
plt.show()

# ---------------------------------------------- PART 6 -------------------------------------------------------

# directory for the files
DIR = 'data/lab_7/potassium'

# read available model files=
files = [f for f in listdir(DIR) if isfile(join(DIR, f)) and not f.startswith('.')]
files = sorted(files)

# load all the files into respective dataframes
background = pd.read_csv(DIR + '/' + files[0])
banana = pd.read_csv(DIR + '/' + files[1])
potato = pd.read_csv(DIR + '/' + files[2])

# subtract background noise from all the data
banana[banana.columns[0]] -= background[background.columns[0]]
potato[potato.columns[0]] -= background[background.columns[0]]

# get area under the curve
# peak occurs at [651:730]
banana_counts = sum(banana.values[651:730])
potato_counts = sum(potato.values[651:730])

# get expected number of counts of each (divide by the efficiency of the detector at that leve * yield)
banana_ec = banana_counts / ((efficiency_of_detector(1460) / 100) * 0.11)
potato_ec = potato_counts / ((efficiency_of_detector(1460) / 100) * 0.11)

# convert the realized counts to the number of decays then to grams
banana_grams = convert_counts_to_grams(banana_ec)
potato_grams = convert_counts_to_grams(potato_ec)

# display the banana information
print('There are {0:.3} g of K in the banana'.format(banana_grams))
print('There are {0:.3} g of K in the potato'.format(potato_grams))
b_percent = (banana_grams / 2975) * 100
p_percent = (potato_grams / 3249) * 100
print("The banana is {0:.3} % K".format(b_percent))
print("The potato is {0:.3} % K".format(p_percent))

# convert all their indices
background.index = convert_axis(background.index)
banana.index = convert_axis(banana.index)
potato.index = convert_axis(potato.index)

# find the peaks and prepare to label them
X, Y = np.linspace(0, len(banana) - 1, len(banana)), banana[banana.columns[0]]
X = convert_axis(X)
peaks, _ = find_peaks(Y, height=np.mean(Y) * 1.5, width=10)

# plot the data
plt.vlines(X[peaks][-1], ymin=0, ymax=max(Y), linestyles='dotted', label='detected peak')
plt.fill_between(X[peaks[-1] - 32: peaks[-1] + 32], 0, max(Y), color='green', alpha=0.25, lw=0, label='area under peak')
plt.plot(X, Y, label='Potassium Banana Trial w/o Background')
plt.title('Banana K Trial')
plt.ylabel('Counts (#)')
plt.xlabel('Energy (keV)')
plt.legend()
plt.figure()

X, Y = np.linspace(0, len(potato) - 1, len(potato)), potato[potato.columns[0]]
X = convert_axis(X)
peaks, _ = find_peaks(Y, height=np.mean(Y) * 1.5, width=10)

# plot the data
plt.vlines(X[peaks][-1], ymin=0, ymax=max(Y), linestyles='dotted', label='detected peak')
plt.fill_between(X[peaks[-1] - 28: peaks[-1] + 40], 0, max(Y), color='green', alpha=0.25, lw=0, label='area under peak')
plt.plot(X, Y, label='Potassium Potato Trial w/o Background')
plt.title('Potato K Trial')
plt.ylabel('Counts (#)')
plt.xlabel('Energy (keV)')
plt.legend()
plt.show()
