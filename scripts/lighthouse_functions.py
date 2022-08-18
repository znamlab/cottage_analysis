import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
import pandas as pd
import math
import statistics as stats
import numpy as np

def obtain_transform_matrix(calibration_directory, calibration_session, x_axis, y_axis, reference, calib_length):

    #The function assumes that you give it three points in space. One is an origin of coordinates and the other two are placed
    #at equal distances from the origin in two perpendicular axes. The distance that separates them is calib_length

    data_ref = pd.read_csv(
        calibration_directory + calibration_session + reference)
    data_x = pd.read_csv(
        calibration_directory + calibration_session + x_axis)
    data_y = pd.read_csv(
        calibration_directory + calibration_session + y_axis)

    light_posit_ref = np.array(
        [stats.mean(data_ref.iloc[:, 2]), stats.mean(data_ref.iloc[:, 3]), stats.mean(data_ref.iloc[:, 4])])
    light_posit_x = np.array(
        [stats.mean(data_x.iloc[:, 2]), stats.mean(data_x.iloc[:, 3]), stats.mean(data_x.iloc[:, 4])])
    light_posit_y = np.array(
        [stats.mean(data_y.iloc[:, 2]), stats.mean(data_y.iloc[:, 3]), stats.mean(data_y.iloc[:, 4])])

    light_posit_refx = light_posit_x - light_posit_ref
    light_posit_refy = light_posit_y - light_posit_ref

    light_posit_refortho = np.cross(light_posit_refx, light_posit_refy)
    norm_fact = (calib_length * np.linalg.norm(light_posit_refx)) / np.linalg.norm(light_posit_refortho)

    light_posit_refortho_scaled = light_posit_refortho * norm_fact
    light_posit_ortho = light_posit_ref + light_posit_refortho_scaled

    light_posit_x, light_posit_y, light_posit_ref, light_posit_ortho = list(
        map(np.append, [light_posit_x, light_posit_y, light_posit_ref, light_posit_ortho], [1]*4))

    # The orthogonal vectors point upwards in the arena, I think it makes sense, check with Antonin

    aruc_posit_ref = np.array([0, 0, 0, 1])
    aruc_posit_x = np.array([calib_length, 0, 0, 1])
    aruc_posit_y = np.array([0, calib_length, 0, 1])
    aruc_posit_ortho = np.cross(aruc_posit_x[0:3], aruc_posit_y[0:3])
    aruc_posit_ortho = np.append(aruc_posit_ortho, 1)

    aruc_matrix = np.transpose([aruc_posit_ref, aruc_posit_x, aruc_posit_y, aruc_posit_ortho])
    light_matrix = np.transpose([light_posit_ref, light_posit_x, light_posit_y, light_posit_ortho])

    # The result is obtained by multiplying the desired set by the inverted original set

    transform_matrix = np.matmul(aruc_matrix, np.linalg.inv(light_matrix))
    return transform_matrix

def obtain_antonin_matrix(calibration_directory, calibration_session, x_axis, y_axis, reference):
    # we now try to do it the Antonin way

    # Get the data

    data_ref = pd.read_csv(
        calibration_directory + calibration_session + reference)
    data_x = pd.read_csv(
        calibration_directory + calibration_session + x_axis)
    data_y = pd.read_csv(
        calibration_directory + calibration_session + y_axis)

    light_posit_ref = np.array(
        [stats.mean(data_ref.iloc[:, 2]), stats.mean(data_ref.iloc[:, 3]), stats.mean(data_ref.iloc[:, 4])])
    light_posit_x = np.array(
        [stats.mean(data_x.iloc[:, 2]), stats.mean(data_x.iloc[:, 3]), stats.mean(data_x.iloc[:, 4])])
    light_posit_y = np.array(
        [stats.mean(data_y.iloc[:, 2]), stats.mean(data_y.iloc[:, 3]), stats.mean(data_y.iloc[:, 4])])


    # Substract the origin from the lighthouse points

    cent_posit_x = light_posit_x - light_posit_ref
    cent_posit_y = light_posit_y - light_posit_ref

    # Compute the cross product for the z pointing vector

    cent_posit_z = np.cross(cent_posit_x, cent_posit_y)

    # Normalization factor to fix the ratio between z and c

    norm_fact = (30 * np.linalg.norm(cent_posit_x)) / np.linalg.norm(cent_posit_z)

    cent_posit_z_scaled = cent_posit_z * norm_fact

    # Divide each vector by their magnitudes: 30, 30, 900

    norm_posit_x = cent_posit_x / 30
    norm_posit_y = cent_posit_y / 30
    norm_posit_z = cent_posit_z_scaled / 900

    # create the transformation matrix

    lin_trans_mat = np.linalg.inv(np.transpose([norm_posit_x, norm_posit_y, norm_posit_z]))

    return lin_trans_mat

def transform_antonin_data(data, transform_matrix, calibration_directory, calibration_session, reference):
    trans_data = data
    data_ref = pd.read_csv(
        calibration_directory + calibration_session + reference)
    light_posit_ref = np.array(
        [stats.mean(data_ref.iloc[:, 2]), stats.mean(data_ref.iloc[:, 3]), stats.mean(data_ref.iloc[:, 4])])

    for index, row in data.iterrows():
        point_vector = np.array([row[2], row[3], row[4]])
        translated_point_vector = point_vector-light_posit_ref
        trans_point = np.matmul(transform_matrix, np.transpose(translated_point_vector))
        trans_data.iloc[index, 2], trans_data.iloc[index, 3], trans_data.iloc[index, 4] = trans_point[0], trans_point[1],trans_point[2]

    return trans_data


def transform_data(data, transform_matrix):
    trans_data = data

    for index, row in data.iterrows():
        point_vector = np.array([row[2], row[3], row[4]])
        point_vector = np.append(point_vector, 1)
        trans_point = np.matmul(transform_matrix, np.transpose(point_vector))
        trans_data.iloc[index, 2], trans_data.iloc[index, 3], trans_data.iloc[index, 4] = trans_point[0], trans_point[1],trans_point[2]

    return trans_data

def plot_single_occupancy(data, colormap):

    #data stands for the output of a lighthouse diode read from the csv as a pandas dataframe
    #colormap is a boolean that specifies whether it should plot the extra dimension of the plot as color

    plt.close('all')

    plt.figure(figsize=(6.8, 4.2))
    if colormap != True:
        plt.scatter(data.iloc[:, 2], data.iloc[:, 3], alpha=0.5, s=0.5)
    else:
        plt.scatter(data.iloc[:, 2], data.iloc[:, 3], c=data.iloc[:, 4], cmap="magma", s=0.5)
        plt.colorbar()
    plt.title("Lighthouse projection of occupancy in the x,y plane")
    plt.xlabel("X axis")
    plt.ylabel("Y axis")

    plt.figure(figsize=(6.8, 4.2))
    if colormap != True:
        plt.scatter(data.iloc[:, 3], data.iloc[:, 4], alpha=0.5, s=0.5)
    else:
        plt.scatter(data.iloc[:, 3], data.iloc[:, 4], c=data.iloc[:, 4], cmap="magma", s=0.5)
        plt.colorbar()
    plt.title("Lighthouse projection of occupancy in the y,z plane")
    plt.xlabel("Y axis")
    plt.ylabel("Z axis")

    plt.figure(figsize=(6.8, 4.2))
    if colormap != True:
        plt.scatter(data.iloc[:, 2], data.iloc[:, 4], alpha=0.5, s=0.5)
    else:
        plt.scatter(data.iloc[:, 2], data.iloc[:, 4], c=data.iloc[:, 4], cmap="magma", s=0.5)
        plt.colorbar()
    plt.title("Lighthouse projection of occupancy in the x,z plane")
    plt.xlabel("X axis")
    plt.ylabel("Z axis")
