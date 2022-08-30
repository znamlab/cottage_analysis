# This is a scratch to load the spike data of the relevant units and
# then plotting a histogram of the units on the freely moving arena.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cottage_analysis.io_module import onix
from scripts import lighthouse_functions as light
from scripts import spike_plotting_functions as spk

spike_clusters = np.load("/Users/colasa/code/SpikeSorting/S20220817/kilosort_int16_converted/spike_clusters.npy")
# It's a vector that gives the cluster to which each spike belongs.
spike_times = np.load("/Users/colasa/code/SpikeSorting/S20220817/kilosort_int16_converted/spike_times.npy")
# Its a vector that gives the time of each spike in samples.

data = onix.load_rhd2164('/Volumes/lab-znamenskiyp/data/instruments/raw_data/projects/blota_onix_pilote/BRAC6692.4a/S20220817/R171408')
clock = np.array(data['clock'][0])

good = [263, 269, 274, 276, 62, 81]

filteredtimes = []
filteredclusters = []
filteredclock = []

for cluster in good:
    mask = (spike_clusters == cluster)
    w_filteredtimes = spike_times[mask]
    w_filteredtimes = np.ndarray.tolist(w_filteredtimes)
    w_filteredclusters = [cluster] * len(w_filteredtimes)
    w_filteredclock = np.arange(len(w_filteredtimes))
    w_filteredclock = clock[w_filteredtimes]
    filteredtimes = np.append(filteredtimes, w_filteredtimes)
    filteredclusters = np.append(filteredclusters, w_filteredclusters)
    filteredclock = np.append(filteredclock, w_filteredclock)

clockinseconds = filteredclock/250000000

calibration_directory = "/Volumes/lab-znamenskiyp/data/instruments/raw_data/projects/blota_onix_pilote/BRAC6692.4a/"
calibration_session = "S20220808"
reference, y_axis, x_axis = ["/calibration_reference/ts4231-1_2022-08-08T16_31_57.csv",
                             "/calibration_xaxis/ts4231-1_2022-08-08T16_33_05.csv",
                             "/calibration_yaxis/ts4231-1_2022-08-08T16_33_54.csv"]

transform = light.obtain_transform_matrix(calibration_directory, calibration_session, x_axis, y_axis, reference,
                                          calib_length=30)

diode_data = pd.read_csv('/Volumes/lab-znamenskiyp/data/instruments/raw_data/projects/blota_onix_pilote/BRAC6692.4a/S20220817/R171408/ts4231-2_2022-08-17T17_14_08.csv')

fixed_data = diode_data

transformed_position = light.transform_data(diode_data, transform)

transformed_position['clockinseconds'] = transformed_position.iloc[:,1]/250000000

cutoff=transformed_position['clockinseconds'].min()

#diff histogram

timelag=list(np.diff(transformed_position['clockinseconds']))
timelag=timelag.append(np.array(timelag).mean())
transformed_position['timelag']=timelag



#Get each unit

unit=81

unitmask = filteredclusters == unit
unit_clockinseconds = clockinseconds[unitmask]

unit_clockinseconds=unit_clockinseconds[unit_clockinseconds>cutoff]

unit_presence = np.arange(len(transformed_position['clockinseconds']))
count = -1
for position in transformed_position['clockinseconds']:
    count = count+1
    if len(list(filter(lambda x: (x>(position-0.2) and x<(position+0.2)), unit_clockinseconds)))>0:
        unit_presence[count] = True

transformed_position['presence']=unit_presence
light.plot_single_occupancy(transformed_position[transformed_position['presence']==True], True)

#Checking the homogeneous spaces between lighthouse datapoints hypothesis:

plt.figure(figsize=(6.8, 4.2))
a = np.diff(transformed_position['clockinseconds'], n=1)
plt.hist(a, log=True)
plt.show()
plt.title('Frequency of delay between consecutive Lighthouse datapoints')
plt.ylabel('Log10(frequency)')
plt.xlabel('Delay (s)')

#Creating the histogram

#First, general occupancy histogram

complete_hist = plt.hist2d(transformed_position.iloc[:,2], transformed_position.iloc[:,3],
                       bins=[100,100], range=[[-40, 40], [-40, 40]], cmmap='magma')
unit_hist = plt.hist2d(transformed_position.iloc[:,2], transformed_position.iloc[:,3],
                       bins=[100,100], range=[[-40, 40], [-40, 40]], cmmap='magma')
print('hello')

#histogram of delay

timelag=list(np.diff(transformed_position['clockinseconds']))
timelag.append(np.array(timelag).mean())
transformed_position['timelag']=timelag

plt.figure(figsize=(6.8, 4.2))
plt.scatter(transformed_position.iloc[:,2], transformed_position.iloc[:,3], c = transformed_position.iloc[:,7], cmap = "magma", s = 2)
plt.title("Delay in space")
plt.xlim(-40, 40)
plt.ylim(-40, 40)
plt.xlabel("X axis (cm)")
plt.ylabel("Y axis (cm)")
plt.colorbar()