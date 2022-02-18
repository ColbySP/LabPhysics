# imports
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from os.path import isfile, join
from os import listdir
import pandas as pd
import numpy as np

# directory for the files
DIR = 'data/cap_lab/plastic_gap'

# read available model files=
files = [f for f in listdir(DIR) if isfile(join(DIR, f)) and not f.startswith('.')]
files = sorted(files)
print('found', files, '\n')

labels = ['91k', '200k', '300k', '510k', '1000k']
labels_int = [91000, 200000, 300000, 510000, 1000000]

# found from the calibration tests
factor = 3.11e-11

# ----------------------------------------- FIRST GAP DISTANCE ------------------------------------
first = []
for i, data in enumerate(files[:5]):
    # load the data
    trial = pd.read_csv(DIR + '/' + data)

    # fix any NaN values that somehow got into our data
    trial['log_voltage'] = np.log(trial['Ch 1  (V)']).fillna(method='ffill')

    # instantiate the regression then fit it
    reg = LinearRegression(fit_intercept=True)
    reg.fit(X=trial['Time(s)'].values.reshape(-1, 1), y=trial['log_voltage'])

    # save regression constants
    slope, intercept = reg.coef_[0], reg.intercept_

    # calculate epsilon_0 from the results
    c_total = -1 / (labels_int[i] * slope)
    first.append(c_total)

    # plot log voltage values
    plt.plot(trial['log_voltage'], label=labels[i])

plt.title('5 Plastic Sheets Trials')
plt.ylabel('Log-Intensity (V)')
plt.xlabel('Time (s)')
plt.legend()
plt.figure()

# ----------------------------------------- SECOND GAP DISTANCE ------------------------------------
second = []
for i, data in enumerate(files[5:10]):
    # load the data
    trial = pd.read_csv(DIR + '/' + data)

    # fix any NaN values that somehow got into our data
    trial['log_voltage'] = np.log(trial['Ch 1  (V)']).fillna(method='ffill')

    # instantiate the regression then fit it
    reg = LinearRegression(fit_intercept=True)
    reg.fit(X=trial['Time(s)'].values.reshape(-1, 1), y=trial['log_voltage'])

    # save regression constants
    slope, intercept = reg.coef_[0], reg.intercept_

    # calculate epsilon_0 from the results
    c_total = -1 / (labels_int[i] * slope)
    second.append(c_total)

    # plot log voltage values
    plt.plot(trial['log_voltage'], label=labels[i])

plt.title('10 Plastic Sheet Trials')
plt.ylabel('Log-Intensity (V)')
plt.xlabel('Time (s)')
plt.legend()
plt.figure()

# ----------------------------------------- THIRD GAP DISTANCE ------------------------------------
third = []
for i, data in enumerate(files[10:15]):
    # load the data
    trial = pd.read_csv(DIR + '/' + data)

    # fix any NaN values that somehow got into our data
    trial['log_voltage'] = np.log(trial['Ch 1  (V)']).fillna(method='ffill')

    # instantiate the regression then fit it
    reg = LinearRegression(fit_intercept=True)
    reg.fit(X=trial['Time(s)'].values.reshape(-1, 1), y=trial['log_voltage'])

    # save regression constants
    slope, intercept = reg.coef_[0], reg.intercept_

    # calculate epsilon_0 from the results
    c_total = -1 / (labels_int[i] * slope)
    third.append(c_total)

    # plot log voltage values
    plt.plot(trial['log_voltage'], label=labels[i])

plt.title('15 Plastic Sheet Trials')
plt.ylabel('Log-Intensity (V)')
plt.xlabel('Time (s)')
plt.legend()

# run regression on c_totals to get capacitance of the compactor vs internal capacitance
reg = LinearRegression(fit_intercept=True)
reg.fit(X=np.reciprocal(np.array([0.006667, 0.013334, 0.02])).reshape(-1, 1), y=[np.mean(first), np.mean(second), np.mean(third)])
slope, internal_capacitance = reg.coef_[0], reg.intercept_

# plt.plot(np.reciprocal(np.array([0.0005, 0.001, 0.0015, 0.002])).reshape(-1, 1), [np.mean(first), np.mean(second), np.mean(third), np.mean(fourth)])

# print results and show final plots
print("Best Estimate of Internal Capacitance C_0: {:.3},".format(internal_capacitance))
print("Best Estimate of E_0: {:.3},".format(slope / 0.04087))  # area of the plates is 12.5cm ^ 2
plt.show()

