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
import lighthouse_functions as lif

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

#2D histogram technique

plt.close('all')

plt.figure(figsize=(6.8, 4.2))
complete_hist = plt.hist2d(data_1.iloc[:,2], data_1.iloc[:,3], bins = [100,100], cmap = "magma")
plt.title('Occupancy histogram of the arena')
plt.colorbar()
plt.figure(figsize=(6.8, 4.2))
delayed_hist = plt.hist2d(data_1_delayed.iloc[:,2], data_1_delayed.iloc[:,3], bins = [100, 100], cmap = "magma")
plt.title('Delayed points [log(cycles)>17] histogram of the arena')
plt.colorbar()

ratio_hist = np.divide(delayed_hist[0], complete_hist[0], out=np.zeros_like(delayed_hist[0]), where=complete_hist[0]!=0)

plt.figure(figsize=(6.8, 4.2))
plt.imshow(ratio_hist, cmap = "magma")
plt.title('Ratio of delayed points to total points histogram of the arena')
plt.colorbar()

###################################################################

#Objective accomplished, but so far, no relevant insights, perhaps divide in a tesselation and calculate average delay?

#We first read the six csvs and create the two relevant matrices: the ARUCO one and the lighthouse one

plt.close('all')

calibration_directory = "/Users/antoniocolas/code/cottage_rawdata/lighthouse_calibration/"
calibration_session = "S20220808"

data_ref = pd.read_csv(calibration_directory+calibration_session+"/calibration_reference/ts4231-1_2022-08-08T16_31_57.csv")
data_x = pd.read_csv(calibration_directory+calibration_session+"/calibration_xaxis/ts4231-1_2022-08-08T16_33_05.csv")
data_y = pd.read_csv(calibration_directory+calibration_session+"/calibration_yaxis/ts4231-1_2022-08-08T16_33_54.csv")

plt.figure(figsize=(6.8, 4.2))
plt.scatter(data_1.iloc[:, 2], data_1.iloc[:, 3], alpha=0.5, s=0.2, c="gray")
plt.scatter(data_ref.iloc[:, 2], data_ref.iloc[:, 3], alpha=1, s=3, c="red")
plt.scatter(data_x.iloc[:, 2], data_x.iloc[:, 3], alpha=1, s=3, c="green")
plt.scatter(data_y.iloc[:, 2], data_y.iloc[:, 3], alpha=1, s=3, c="blue")
plt.title("Lighthouse projection of occupancy in the x,y plane")
plt.xlabel("X axis")
plt.ylabel("Y axis")

light_posit_ref = np.array([stats.mean(data_ref.iloc[:,2]), stats.mean(data_ref.iloc[:,3]), stats.mean(data_ref.iloc[:,4]), 1])
light_posit_x = np.array([stats.mean(data_x.iloc[:,2]), stats.mean(data_x.iloc[:,3]), stats.mean(data_x.iloc[:,4]), 1])
light_posit_y = np.array([stats.mean(data_y.iloc[:,2]), stats.mean(data_y.iloc[:,3]), stats.mean(data_y.iloc[:,4]), 1])

light_posit_ref = np.array(
    [stats.mean(data_ref.iloc[:, 2]), stats.mean(data_ref.iloc[:, 3]), stats.mean(data_ref.iloc[:, 4])])
light_posit_x = np.array(
    [stats.mean(data_x.iloc[:, 2]), stats.mean(data_x.iloc[:, 3]), stats.mean(data_x.iloc[:, 4])])
light_posit_y = np.array(
    [stats.mean(data_y.iloc[:, 2]), stats.mean(data_y.iloc[:, 3]), stats.mean(data_y.iloc[:, 4])])

light_posit_refx = light_posit_x - light_posit_ref
light_posit_refy = light_posit_y - light_posit_ref

light_posit_refortho = np.cross(light_posit_refx, light_posit_refy)
light_posit_ortho = light_posit_ref + light_posit_refortho

light_posit_x, light_posit_y, light_posit_ref, light_posit_ortho = list(
    map(np.append, [light_posit_x, light_posit_y, light_posit_ref, light_posit_ortho], [1]*4))

# The orthogonal vectors point upwards in the arena, I think it makes sense, check with Antonin

aruc_posit_ref = np.array([0, 0, 0, 1])
aruc_posit_x = np.array([30, 0, 0, 1])
aruc_posit_y = np.array([0, 30, 0, 1])
# Because the magnitude of a cross product is the area of the paralellogram that has the terms of the product as
# sides, and posit_x and posit_y are perpendicular, then the magnitude of posit_ortho is 30^2=900
aruc_posit_ortho = np.array([0, 0, 900, 1])

aruc_matrix = np.transpose([aruc_posit_ref, aruc_posit_x, aruc_posit_y, aruc_posit_ortho])
light_matrix = np.transpose([light_posit_ref, light_posit_x, light_posit_y, light_posit_ortho])

# The result is obtained by multiplying the desired set by the inverted original set

transform_matrix = np.matmul(aruc_matrix, np.linalg.inv(light_matrix))

#When you multiply a lighthouse position vector by the transform matrix you should get the aruco coordinates

#Let's test that.

np.matmul(transform_matrix, np.transpose(light_posit_x))

#we now try to do it the Antonin way

#Get the data

light_posit_ref = np.array([stats.mean(data_ref.iloc[:,2]), stats.mean(data_ref.iloc[:,3]), stats.mean(data_ref.iloc[:,4])])
light_posit_x = np.array([stats.mean(data_x.iloc[:,2]), stats.mean(data_x.iloc[:,3]), stats.mean(data_x.iloc[:,4])])
light_posit_y = np.array([stats.mean(data_y.iloc[:,2]), stats.mean(data_y.iloc[:,3]), stats.mean(data_y.iloc[:,4])])

light_posit_ref = np.array(
    [stats.mean(data_ref.iloc[:, 2]), stats.mean(data_ref.iloc[:, 3]), stats.mean(data_ref.iloc[:, 4])])
light_posit_x = np.array(
    [stats.mean(data_x.iloc[:, 2]), stats.mean(data_x.iloc[:, 3]), stats.mean(data_x.iloc[:, 4])])
light_posit_y = np.array(
    [stats.mean(data_y.iloc[:, 2]), stats.mean(data_y.iloc[:, 3]), stats.mean(data_y.iloc[:, 4])])

#Substract the origin from the lighthouse points

cent_posit_x = light_posit_x-light_posit_ref
cent_posit_y = light_posit_y-light_posit_ref


#Compute the cross product for the z pointing vector

cent_posit_z = np.cross(cent_posit_y, cent_posit_x)

#Normalization factor to fix the ratio between z and c

norm_fact = (30*np.linalg.norm(cent_posit_x))/np.linalg.norm(cent_posit_z)

cent_posit_z_scaled = cent_posit_z*norm_fact

plt.close('all')

plt.figure(figsize=(6.8, 4.2))
plt.scatter(data_1.iloc[:, 2], data_1.iloc[:, 3], alpha=0.5, s=0.2, c="gray")
plt.scatter(data_ref.iloc[:, 2], data_ref.iloc[:, 3], alpha=1, s=3, c="red")
plt.scatter(data_x.iloc[:, 2], data_x.iloc[:, 3], alpha=1, s=3, c="green")
plt.scatter(data_y.iloc[:, 2], data_y.iloc[:, 3], alpha=1, s=3, c="blue")
plt.title("Intact lighthouse projection of occupancy in the x,y plane")
plt.xlabel("X axis")
plt.ylabel("Y axis")

plt.figure(figsize=(6.8, 4.2))
plt.scatter(data_1.iloc[:, 2], data_1.iloc[:, 3], alpha=0.5, s=0.2, c="gray")
plt.scatter(0, 0, alpha=1, s=3, c="red")
plt.scatter(cent_posit_x[0], cent_posit_x[1], alpha=1, s=3, c="green")
plt.scatter(cent_posit_y[0], cent_posit_y[1], alpha=1, s=3, c="blue")
plt.scatter(cent_posit_z[0], cent_posit_z[1], alpha=1, s=3, c="purple")
plt.title("Lighthouse projection of occupancy in the x,y plane")
plt.xlabel("X axis")
plt.ylabel("Y axis")

plt.figure(figsize=(6.8, 4.2))
plt.scatter(data_1.iloc[:, 2], data_1.iloc[:, 4], alpha=0.5, s=0.2, c="gray")
plt.scatter(0, 0, alpha=1, s=3, c="red")
plt.scatter(cent_posit_x[1], cent_posit_x[2], alpha=1, s=3, c="green")
plt.scatter(cent_posit_y[1], cent_posit_y[2], alpha=1, s=3, c="blue")
plt.scatter(cent_posit_z[1], cent_posit_z[2], alpha=1, s=3, c="purple")
plt.title("Lighthouse projection of occupancy in the x,z plane")
plt.xlabel("X axis")
plt.ylabel("Z axis")

#Divide each vector by their magnitudes: 30, 30, 900

norm_posit_x = cent_posit_x/30
norm_posit_y = cent_posit_y/30
norm_posit_z = cent_posit_z_scaled/900

#create the transformation matrix



lin_trans_mat = np.linalg.inv(np.transpose([norm_posit_x, norm_posit_y, norm_posit_z]))



#Now we transform every point of the recording and repeat the plots.

calibration_directory = "/Users/antoniocolas/code/cottage_rawdata/lighthouse_calibration/"
calibration_session = "S20220808"

reference, x_axis, y_axis = ["/calibration_reference/ts4231-1_2022-08-08T16_31_57.csv", "/calibration_xaxis/ts4231-1_2022-08-08T16_33_05.csv", "/calibration_yaxis/ts4231-1_2022-08-08T16_33_54.csv"]

tran_matrix = lif.obtain_transform_matrix(calibration_directory, calibration_session, x_axis, y_axis, reference)

#t_data_1, t_data_2 = transform_data(data_1, tran_matrix), transform_data(data_2, tran_matrix)

t_data_1 = lif.transform_data(data_1, tran_matrix)

lif.plot_single_occupancy(t_data_1, colormap=False)

print('hello world')
