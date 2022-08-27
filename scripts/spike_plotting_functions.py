
#These are functions that I've found useful to plot the spike data in spatial coordinates.

import numpy as np
from matplotlib import pyplot as plt



def plot_unit_in_xy(data, clockinseconds, filteredclusters, unit, window, colormap):

    #Data is a pandas dataframe of a lighthouse diode's positions

    #clockinseconds is the vector with clock times of all the spikes you're interested in

    #filteredclusters is a vector that contains the cluster each spike you have belongs to

    #unit is a cluster you're interested in plotting

    #window is the time window before or after the time of the spike you allow a possition to be
    #associated to that spike

    #colormap is a boolean in case you want to color the plot by the z coordinate.


    data['clockinseconds'] = data.iloc[:, 1] / 250000000
    cutoff = data['clockinseconds'].min()

    unitmask = filteredclusters == unit
    unit_clockinseconds = clockinseconds[unitmask]
    unit_clockinseconds = unit_clockinseconds[unit_clockinseconds > cutoff]

    unit_presence = np.arange(len(data['clockinseconds']))
    count = -1
    for position in data['clockinseconds']:
        count = count + 1
        #if len(list(filter(lambda x: (x > (position - window) and x < (position + window)), unit_clockinseconds))) > 0:
        #unit_presence[count] = True
        dum=unit_clockinseconds[unit_clockinseconds<(position+window)]
        dum=dum[dum>(position-window)]
        if len(dum)>0:
            unit_presence[count] = True

    data['presence'] = unit_presence

    plotting_data = data[data['presence']==True]

    plt.figure(figsize=(6.8, 4.2))
    if colormap != True:
        plt.scatter(plotting_data.iloc[:, 2], plotting_data.iloc[:, 3], alpha=0.5, s=0.5)
    else:
        plt.scatter(plotting_data.iloc[:, 2], plotting_data.iloc[:, 3], c=plotting_data.iloc[:, 4], cmap="magma", s=0.5)
        plt.colorbar()
    plt.title("Firing of of unit "+str(unit)+' in the xy plane')
    plt.xlabel("X axis")
    plt.ylabel("Y axis")

def plot_firing_rate_histogram(data, clockinseconds, filteredclusters, unit, window, bins):
    # Data is a pandas dataframe of a lighthouse diode's positions

    # clockinseconds is the vector with clock times of all the spikes you're interested in

    # filteredclusters is a vector that contains the cluster each spike you have belongs to

    # unit is a cluster you're interested in plotting

    # window is the time window before or after the time of the spike you allow a possition to be
    # associated to that spike

    # colormap is a boolean in case you want to color the plot by the z coordinate.

    data['clockinseconds'] = data.iloc[:, 1] / 250000000
    cutoff = data['clockinseconds'].min()

    unitmask = filteredclusters == unit
    unit_clockinseconds = clockinseconds[unitmask]
    unit_clockinseconds = unit_clockinseconds[unit_clockinseconds > cutoff]

    unit_presence = np.arange(len(data['clockinseconds']))
    count = -1
    for position in data['clockinseconds']:
        count = count + 1
        # if len(list(filter(lambda x: (x > (position - window) and x < (position + window)), unit_clockinseconds))) > 0:
        # unit_presence[count] = True
        dum = unit_clockinseconds[unit_clockinseconds < (position + window)]
        dum = dum[dum > (position - window)]
        if len(dum) > 0:
            unit_presence[count] = True

    data['presence'] = unit_presence

    plotting_data = data[data['presence'] == True]

    complete_hist = plt.hist2d(data.iloc[:, 2], data.iloc[:, 3]
                               ,bins=[bins, bins], range=[[-40, 40], [-40, 40]])
    unit_hist = plt.hist2d(plotting_data.iloc[:, 2], plotting_data.iloc[:, 3]
                           ,bins=[bins, bins], range=[[-40, 40], [-40, 40]])
    ratio_hist = np.divide(unit_hist[0], complete_hist[0], out=np.zeros_like(unit_hist[0]), where=complete_hist[0]!=0)
    plt.figure(figsize=(10, 8))
    plt.imshow(ratio_hist, cmap='magma')
    plt.colorbar()
    plt.title('Firing rate histogram of unit '+str(unit)+' window of '+str(window)+'s')