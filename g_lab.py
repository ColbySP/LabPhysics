# imports
from scipy.optimize import curve_fit
from os.path import isfile, join
import matplotlib.pyplot as plt
from os import listdir
import pandas as pd
import numpy as np
import math


# directory for the files
DIR = 'data/g_lab'
L = 2.1660
l = 0.1 / 2
M = 0.499395
H = 0.036


# curve fitting function
def torsion_pendulum(t, a, w, A, phi, c):
    return (A * np.exp(-1 * a * t) * np.cos((w * t) + phi)) + c


# function to calculate G based on constants and fit parameters
def calc_G(l, d, omega, alpha, H, L, M):
    num = l * d * (omega ** 2 + alpha ** 2) * (H - (l * (d / (4*L)))) ** 2
    dom = 4 * L * M
    return num / dom


# function to reindex the voltages to distance
def reindex(x, u, l, d=0.2743):
    factor = d / (u - l)
    return x * factor


# read available csv files
files = [f for f in listdir(DIR) if isfile(join(DIR, f)) and not f.startswith('.')]
files = sorted(files)
print('found', files, '\n')

calibration = pd.read_csv(DIR + '/' + files[0]).set_index('Time(s)')
upper_limit, lower_limit = max(calibration['Position(V)']), min(calibration['Position(V)'])

G_list = []
for index, data in enumerate(files[1:]):
    # load data
    trial = pd.read_csv(DIR + '/' + data).set_index('Time(s)')
    trial['Position(X)'] = reindex(trial['Position(V)'], upper_limit, lower_limit)

    # curve fit the data using the equation
    popt, _ = curve_fit(torsion_pendulum, trial.index, trial['Position(X)'],
                        p0=[0.0005, 0.0095, 0.1, math.pi / 2, 0.04])

    # get variables from the fitting
    a, w, A, phi, c = popt

    # get other important variables for calculation of G
    d = upper_limit - lower_limit

    # calculate G given the new parameters
    G = calc_G(l, d, w, a, H, L, M)
    # print("G is calculated to be: {:.5}".format(G))
    G_list.append(G)

    # draw line of best fit
    x_hat = np.linspace(0, max(trial.index.values), 500)
    y_hat = torsion_pendulum(x_hat, a, w, A, phi, c)

    # plot data
    plt.plot(trial['Position(X)'], label='Collected Data')
    plt.plot(x_hat, y_hat, label='Line Of Best Fit', c='r')
    plt.title('G-Lab Experiment ' + files[index])
    plt.ylabel('Position (X)')
    plt.xlabel('Time (S)')
    plt.legend()
    plt.figure()

# show figures
print("Mean G: {:.3},".format(np.mean(G_list)), "Standard Error: {:.3}".format(np.std(G_list, ddof=1) / np.sqrt(np.size(G_list))))
plt.show()
