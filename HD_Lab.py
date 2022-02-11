from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from os.path import isfile, join
from os import listdir
import pandas as pd
import numpy as np


# directory for the files
DIR = 'data/hd_lab'

# read available model files=
files = [f for f in listdir(DIR) if isfile(join(DIR, f)) and not f.startswith('.')]
files = sorted(files)
print('found', files, '\n')

labels = ['Slit Width 25', 'Slit Width 22', 'Slit Width 20',  'Slit Width 15', 'Slit Width 12']

first = []
for i, data in enumerate(files[:5]):
    # load the data
    trial = pd.read_csv(DIR + '/' + data)

    # load data to numpy arrays
    marks = trial['Marks'].values
    intensity = trial['Intensity(V)'].values

    # find marks and mark differences
    mark_index = np.argwhere(~np.isnan(marks))
    time_delta = max(mark_index) - min(mark_index)
    difference = max(marks[mark_index]) - min(marks[mark_index])

    # find peaks and peak differences
    peaks, _ = find_peaks(intensity, prominence=0.5)
    peak_difference = max(peaks) - min(peaks)

    # calculate delta lambda
    delta_lambda = (difference / time_delta) * peak_difference
    first.append(delta_lambda[0])

    # plot data
    plt.vlines(peaks, ymin=0, ymax=max(intensity), linestyles='dotted')
    plt.plot(trial['Intensity(V)'], label=labels[i % 5])

plt.title('656 NM Trials')
plt.ylabel('Intensity (V)')
plt.xlabel('Time (s)')
plt.legend()
plt.figure()

# ----------------------------------------------------------------------------------------------------

second = []
for i, data in enumerate(files[5:10]):
    # load the data
    trial = pd.read_csv(DIR + '/' + data)

    # load data to numpy arrays
    marks = trial['Marks'].values
    intensity = trial['Intensity(V)'].values

    # find marks and mark differences
    mark_index = np.argwhere(~np.isnan(marks))
    time_delta = max(mark_index) - min(mark_index)
    difference = max(marks[mark_index]) - min(marks[mark_index])

    # find peaks and peak differences
    peaks, _ = find_peaks(intensity, prominence=0.159)
    peak_difference = max(peaks) - min(peaks)

    # calculate delta lambda
    delta_lambda = (difference / time_delta) * peak_difference
    second.append(delta_lambda[0])

    # plot data
    plt.vlines(peaks, ymin=0, ymax=max(intensity), linestyles='dotted')
    plt.plot(trial['Intensity(V)'], label=labels[i % 5])

plt.title('486 NM Trials')
plt.ylabel('Intensity (V)')
plt.xlabel('Time (s)')
plt.legend()
plt.figure()

# ----------------------------------------------------------------------------------------------------

third = []
for i, data in enumerate(files[10:14]):
    # load the data
    trial = pd.read_csv(DIR + '/' + data)

    # load data to numpy arrays
    marks = trial['Marks'].values
    intensity = trial['Intensity(V)'].values

    # find marks and mark differences
    mark_index = np.argwhere(~np.isnan(marks))
    time_delta = max(mark_index) - min(mark_index)
    difference = max(marks[mark_index]) - min(marks[mark_index])

    # find peaks and peak differences
    peaks, _ = find_peaks(intensity, prominence=0.065)
    peak_difference = max(peaks) - min(peaks)

    # calculate delta lambda
    delta_lambda = (difference / time_delta) * peak_difference
    third.append(delta_lambda[0])

    # plot data
    plt.vlines(peaks, ymin=0, ymax=max(intensity), linestyles='dotted')
    plt.plot(trial['Intensity(V)'], label=labels[i % 5])

plt.title('434 NM Trials')
plt.ylabel('Intensity (V)')
plt.xlabel('Time (s)')
plt.legend()
plt.figure()

# ----------------------------------------------------------------------------------------------------

fourth = []
for i, data in enumerate(files[14:18]):
    # load the data
    trial = pd.read_csv(DIR + '/' + data)

    # load data to numpy arrays
    marks = trial['Marks'].values
    intensity = trial['Intensity(V)'].values

    # find marks and mark differences
    mark_index = np.argwhere(~np.isnan(marks))
    time_delta = max(mark_index) - min(mark_index)
    difference = max(marks[mark_index]) - min(marks[mark_index])

    # find peaks and peak differences
    if i == 0:
        peaks = [88, 96]
    elif i == 1:
        peaks = [325, 360]
    elif i == 2:
        peaks = [212, 265]
    else:
        peaks = [70, 92]

    peak_difference = max(peaks) - min(peaks)

    # calculate delta lambda
    delta_lambda = (difference / time_delta) * peak_difference
    fourth.append(delta_lambda[0])

    # plot data
    plt.vlines(peaks, ymin=0, ymax=max(intensity), linestyles='dotted')
    plt.plot(trial['Intensity(V)'], label=labels[i % 5])

plt.title('410 NM Trials')
plt.ylabel('Intensity (V)')
plt.xlabel('Time (s)')
plt.legend()
plt.figure()

# ------------------------------------------------------------------------------------------------------
mystery = []
for i, data in enumerate(files[18:]):
    # load the data
    trial = pd.read_csv(DIR + '/' + data)

    # load data to numpy arrays
    marks = trial['Marks'].values
    intensity = trial['Intensity(V)'].values

    # find marks and mark differences
    mark_index = np.argwhere(~np.isnan(marks))
    time_delta = max(mark_index) - min(mark_index)
    difference = max(marks[mark_index]) - min(marks[mark_index])

    # find peaks and peak differences
    peaks, _ = find_peaks(intensity, prominence=1)

    # calculate delta lambda
    lambdas = ((difference / time_delta) * peaks) + min(marks[mark_index])

    # plot data
    plt.vlines(peaks, ymin=0, ymax=max(intensity), linestyles='dotted')
    plt.plot(trial['Intensity(V)'], label=labels[i % 5])

plt.title('Mystery Gas')
plt.ylabel('Intensity (V)')
plt.xlabel('Time (s)')
plt.legend()

print("Mean Delta Lambda: {:.3},".format(np.mean(first)), "Standard Error: {:.3}".format(np.std(first, ddof=1) / np.sqrt(np.size(first))))
print("Mean Delta Lambda: {:.3},".format(np.mean(second)), "Standard Error: {:.3}".format(np.std(second, ddof=1) / np.sqrt(np.size(second))))
print("Mean Delta Lambda: {:.3},".format(np.mean(third)), "Standard Error: {:.3}".format(np.std(third, ddof=1) / np.sqrt(np.size(third))))
print("Mean Delta Lambda: {:.3},".format(np.mean(fourth)), "Standard Error: {:.3}".format(np.std(fourth, ddof=1) / np.sqrt(np.size(fourth))))

print('\nMystery Material Peaks At Wavelengths:', lambdas)

plt.show()
