"""
Small script to load lighthouse data, align them with arena floor and plot mouse position
"""
import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
import pandas as pd
import math
import statistics as stats
import numpy as np

rawdata_directory = "/Users/antoniocolas/code/cottage_rawdata/BRAC6692.4a/S20220808/"

data_1 = pd.read_csv(rawdata_directory+"open_fieldts4231-1_2022-08-08T16_51_31.csv")
data_2 = pd.read_csv(rawdata_directory+"open_fieldts4231-2_2022-08-08T16_51_31.csv")

#Plotting the projected occupancies in the three relevantt planes

plt.close('all')

plt.figure(figsize=(6.8, 4.2))
plt.scatter(data_1.iloc[:,2], data_1.iloc[:,3], c = data_1.iloc[:,4], cmap = "magma", s = 0.5)
plt.title("Lighthouse projection of occupancy in the x,y plane")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.colorbar()

plt.figure(figsize=(6.8, 4.2))
plt.scatter(data_1.iloc[:,3], data_1.iloc[:,4], c = data_1.iloc[:,2], cmap = "magma", s = 0.5)
plt.title("Lighthouse projection of occupancy in the y,z plane")
plt.xlabel("Y axis")
plt.ylabel("Z axis")
plt.colorbar()

plt.figure(figsize=(6.8, 4.2))
plt.scatter(data_1.iloc[:,2], data_1.iloc[:,4], c = data_1.iloc[:,3], cmap = "magma", s = 0.5)
plt.title("Lighthouse projection of occupancy in the x,z plane")
plt.xlabel("X axis")
plt.ylabel("Z axis")
plt.colorbar()

###############


plt.figure(figsize=(6.8, 4.2))
plt.scatter(data_1.iloc[:,2], data_1.iloc[:,3], c = data_1.iloc[:,4], cmap = "magma", s = 0.5)
plt.scatter(data_2.iloc[:,2], data_2.iloc[:,3], c = data_2.iloc[:,4], cmap = "magma", s = 0.5)
plt.title("two-diode lighthouse projection of occupancy in the x,y plane")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.colorbar()

plt.figure(figsize=(6.8, 4.2))
plt.scatter(data_1.iloc[:,3], data_1.iloc[:,4], c = data_1.iloc[:,2], cmap = "magma", s = 0.5)
plt.scatter(data_2.iloc[:,3], data_2.iloc[:,4], c = data_2.iloc[:,2], cmap = "magma", s = 0.5)
plt.title("two-diode lighthouse projection of occupancy in the y,z plane")
plt.xlabel("Y axis")
plt.ylabel("Z axis")
plt.colorbar()

plt.figure(figsize=(6.8, 4.2))
plt.scatter(data_1.iloc[:,2], data_1.iloc[:,4], c = data_1.iloc[:,3], cmap = "magma", s = 0.5)
plt.scatter(data_2.iloc[:,3], data_2.iloc[:,4], c = data_2.iloc[:,2], cmap = "magma", s = 0.5)
plt.title("two-diode lighthouse projection of occupancy in the x,z plane")
plt.xlabel("X axis")
plt.ylabel("Z axis")
plt.colorbar()

#Calculating the delay between measurements and projecting it.

delay_1 = [0]*len(data_1.iloc[:,1])
logdelay_1 = [0]*len(data_1.iloc[:,1])

for i in range(0,len(data_1.iloc[:,1])):
    if i == 0:
        delay_1[i] = 0
        logdelay_1[i] = None
    else:
        delay_1[i] = data_1.iloc[i,1]-data_1.iloc[(i-1),1]
        logdelay_1[i] = math.log(delay_1[i])

delay_2 = [0]*len(data_2.iloc[:,1])
logdelay_2 = [0]*len(data_2.iloc[:,1])

for i in range(0,len(data_2.iloc[:,1])):
    if i == 0:
        delay_2[i] = 0
        logdelay_2[i] = None
    else:
        delay_2[i] = data_2.iloc[i,1]-data_2.iloc[(i-1),1]
        logdelay_2[i] = math.log(delay_2[i])

plt.figure(figsize = (6.8,4.2))
plt.hist(list(filter(None, logdelay_1)), bins = list(range(14,25, 1)))
plt.yscale('log')
plt.title("Histogram of log(delay)")
plt.ylabel("Absolute frequency")
plt.xlabel("log(delay)")

data_1 = data_1.assign(delay = delay_1)
data_1 = data_1.assign(logdelay = logdelay_1)
data_2 = data_2.assign(delay = delay_2)
data_2 = data_2.assign(logdelay = logdelay_2)

plt.figure(figsize=(6.8, 4.2))
plt.scatter(data_1.iloc[:,2], data_1.iloc[:,3], c = data_1.iloc[:,5], cmap = "magma", s = 0.5)
plt.title("Lighthouse projection of delay in the x,y plane")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.colorbar()

plt.figure(figsize=(6.8, 4.2))
plt.scatter(data_1.iloc[:,2], data_1.iloc[:,3], c = data_1.iloc[:,6], cmap = "magma", s = 2)
plt.title("Lighthouse projection of log delay in the x,y plane")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.colorbar()

data_1_delayed = data_1[data_1.logdelay!=None]
data_1_delayed = data_1_delayed[data_1_delayed.logdelay>17]

plt.figure(figsize=(6.8, 4.2))
plt.scatter(data_1_delayed.iloc[:,2], data_1_delayed.iloc[:,3], c = data_1_delayed.iloc[:,6], cmap = "magma", s = 2)
plt.title("Lighthouse projection of log delay of logdelay>17 points in the x,y plane")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.colorbar()

#Objective accomplished, but so far, no relevant insights, perhaps divide in a tesselation and calculate average delay?

#We first read the six csvs and create the two relevant matrices: the ARUCO one and the lighthouse one

plt.close('all')

calibration_directory = "/Users/antoniocolas/code/cottage_rawdata/lighthouse_calibration/"
calibration_session = "S20220808"

data_ref = pd.read_csv(calibration_directory+calibration_session+"/calibration_reference/ts4231-1_2022-08-08T16_31_57.csv")
data_x = pd.read_csv(calibration_directory+calibration_session+"/calibration_xaxis/ts4231-1_2022-08-08T16_33_05.csv")
data_y = pd.read_csv(calibration_directory+calibration_session+"/calibration_yaxis/ts4231-1_2022-08-08T16_33_54.csv")

light_posit_ref = [stats.mean(data_ref.iloc[:,2]), stats.mean(data_ref.iloc[:,3]), stats.mean(data_ref.iloc[:,4])]
light_posit_x = [stats.mean(data_x.iloc[:,2]), stats.mean(data_x.iloc[:,3]), stats.mean(data_x.iloc[:,4])]
light_posit_y = [stats.mean(data_y.iloc[:,2]), stats.mean(data_y.iloc[:,3]), stats.mean(data_y.iloc[:,4])]
light_posit_ortho = [0]*3

for dimension in [1,2,3]:
    if dimension+1 == 4:
        index_1 = 1
    else:
        index_1 = dimension+1
    if dimension-1 == -1:
        index_2 = 3
    else:
        index_2 = dimension-1
    light_posit_ortho[dimension-1] = (light_posit_x[index_1-1]*light_posit_y[index_2-1])-(light_posit_x[index_2-1]*light_posit_y[index_2-1])

aruc_posit_ref = [0, 0, 0]
aruc_posit_x = [30, 0, 0]
aruc_posit_y = [0, 30, 0]
aruc_posit_ortho = [0, 0, 30]

aruc_matrix = np.transpose(np.array([aruc_posit_x, aruc_posit_y, aruc_posit_ortho]))
light_matrix = np.transpose(np.array([light_posit_x, light_posit_y, light_posit_ortho]))

#The result is obtained by multiplying the desired set by the inverted original set



#light_vect_refx = [0]*3
#light_vect_refy = [0]*3
#light_vect_ortho = [0]*3

#for dimension in [0, 1, 2]:
    #light_vect_refx = light_posit_x[dimension]-light_posit_ref[dimension]
    #light_vect_refy = light_posit_y[dimension]-light_posit_ref[dimension]
   # light_vect_ortho = light_vect


#light_vect_refy =
#light_vect_ortho =