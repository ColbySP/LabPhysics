# imports
from sklearn.linear_model import LinearRegression
from scipy.signal import find_peaks
from os.path import isfile, join
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.stats import sem
from os import listdir
import pandas as pd
import numpy as np
import scipy
import math

# directory for the files
DIR = 'data/lz_lab'


# Define the Gaussian function
def gaussian(x, amp1, cen1, sigma1):
    return amp1 * (1 / (sigma1 * (np.sqrt(2 * np.pi)))) * (np.exp((-1.0 / 2.0) * (((x - cen1) / sigma1) ** 2)))


# read available model files=
files = [f for f in listdir(DIR) if isfile(join(DIR, f)) and not f.startswith('.')]
files = sorted(files)
print('found', files, '\n')

# peak 1 @ time 0.033400
# peak 2 @ time 0.076400
# found from looking at ZE_220317_001.csv
peak1, peak2 = 0.033400, 0.076400
difference = (peak2 - peak1) / 8  # one difference is equal to 8Ghz so we divide by 8

# collected peaks data
spacings = []
pzt = []
for i, data in enumerate(files[3:6]):
    # load the data
    trial = pd.read_csv(DIR + '/' + data).set_index('Time(s)')
    trial.drop('Unnamed: 3', axis=1, inplace=True)
    trial['Ghz'] = trial.index / difference

    # get rid of the bad ends of the data collection and resample data
    tails = 60
    X, Y, Z = trial['Ghz'].values[:-tails], trial['Ch 1  (V)'].values[:-tails],  trial['Ch 2  (V)'].values[:-tails]

    # find peaks and peak differences
    peaks, _ = find_peaks(Y, distance=10, prominence=0.02)
    # subsample for the small
    peaks = peaks[-2:]

    # get the change in frequency per volt
    space = abs(X[peaks][0] - X[peaks][1])
    volt = abs(Z[peaks][0] - Z[peaks][1])
    pzt.append(633 / (2 * volt))

    # get the spacing between the large and small peak
    spacings.append(space)

    # add the data to the figure
    plt.plot(X, (Y - min(Y)) / (max(Y) - min(Y)), label='data', c='k')
    plt.plot(X, (Z - min(Z)) / (max(Z) - min(Z)), label='voltage', c='r')
    plt.vlines(X[peaks], ymin=0, ymax=max(Z), linestyles='dotted', label='detected peak')

    plt.title('Regular Trial: ' + data)
    plt.xlabel('Frequency (Ghz)')
    plt.ylabel('Normalized Intensity (V)')
    plt.figure()

# gain curve trials
fwhm = []
fastest_atoms = []
for i, data in enumerate(files[:3]):
    # load the data
    trial = pd.read_csv(DIR + '/' + data).set_index('Time(s)')
    trial.drop('Unnamed: 3', axis=1, inplace=True)
    trial['Ghz'] = trial.index / difference

    # get rid of the bad ends of the data collection and resample data
    tails = 100
    X, Y = trial['Ghz'].values[tails + 1:-tails:2], trial['Ch 1  (V)'].values[tails + 1:-tails:2]
    Y = Y - min(Y)

    # fit a gaussian curve
    popt_gauss, pcov_gauss = scipy.optimize.curve_fit(gaussian, X, Y, p0=[1, 9, 1])
    amp, mu, std = popt_gauss

    # calculate the full width half maximum
    width = 2 * np.sqrt(2 * (math.log(2))) * std
    fwhm.append(width)

    # get the fastest lasing atoms
    fastest_atoms.append((width * 300000000) / mu)

    # add the data to the figure
    plt.scatter(X, Y, label='data', c='k')
    plt.plot(X, gaussian(x=X, amp1=amp, cen1=mu, sigma1=std), label='fit', c='r')

    # add labels to the figure
    plt.title('Gain Curve Trial: ' + data)
    plt.xlabel('Frequency (Ghz)')
    plt.ylabel('Intensity (V)')
    plt.legend()
    plt.figure()

# b field trials
differences = []
for i, data in enumerate(files[6:]):
    # load the data
    trial = pd.read_csv(DIR + '/' + data).set_index('Time(s)')
    trial.drop('Unnamed: 3', axis=1, inplace=True)
    trial['Ghz'] = trial.index / difference

    # get rid of the bad ends of the data collection and resample data
    tails = 60
    X, Y = trial['Ghz'].values[:-tails], trial['Ch 1  (V)'].values[:-tails]

    # find peaks and peak differences
    peaks, _ = find_peaks(Y, distance=50, prominence=0.01)

    # add the data to the figure
    plt.plot(X, Y, label='data', c='k')
    plt.vlines(X[peaks], ymin=0, ymax=max(Y), linestyles='dotted', label='detected peak')

    differences.append(((X[peaks][2] - X[peaks][0]) + (X[peaks][3] - X[peaks][1])) / 2)

    # add labels to the figure
    plt.title('B-Field Trial: ' + data)
    plt.xlabel('Frequency (Ghz)')
    plt.ylabel('Intensity (V)')
    plt.legend()
    plt.figure()

# fit linear regression to the zeeman effect and b-field strength
Bfield = [0.927, 1.014, 1.1, 1.178, 1.258, 1.326, 1.060, 1.255, 0.945, 1.111, 1.257]
reg = LinearRegression().fit(np.array(Bfield).reshape(-1, 1), np.array(differences).reshape(-1, 1))

plt.scatter(x=Bfield, y=differences, label='collected B-field data', c='k')
plt.plot(np.linspace(min(Bfield), max(Bfield)).reshape(-1, 1), (np.linspace(min(Bfield), max(Bfield)) * reg.coef_ + reg.intercept_).reshape(-1, 1), label='line of best fit', c='r')
plt.xlabel('B-field strength (kG)')
plt.ylabel('delta f difference (Ghz)')
plt.title('change in frequency vs. B-field strength')
plt.legend()

print('the laser mode spacing is: {:.3f}Ghz'.format(np.mean(spacings)))
print('standard error was, {:.3f}'.format(sem(spacings)))
print('full width half maximum is: {:.3f}Ghz'.format(np.mean(fwhm)))
print('standard error was, {:.3f}'.format(sem(fwhm)))
print('fastest lasing atoms are traveling: {:.0f}m/s'.format(np.mean(fastest_atoms)))
print('standard error was, {:.3f}'.format(sem(fastest_atoms)))
print('pzt length change per volt is: {:.3f}'.format(np.mean(pzt)))
print('standard error was, {:.3f}'.format(sem(pzt)))
print('zeeman splitting coefficient is: {:.3f}'.format(reg.coef_[0][0]))
print('laser cavity length from room temp to operating temp is: {:.3f}nm'.format((633 / 2) * 91))

plt.show()
