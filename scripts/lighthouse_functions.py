import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
import pandas as pd
import math
import statistics as stats
import numpy as np

def obtain_transform_matrix(calibration_directory, calibration_session, x_axis, y_axis, reference):

    data_ref = pd.read_csv(
        calibration_directory + calibration_session + reference)
    data_x = pd.read_csv(
        calibration_directory + calibration_session + y_axis)
    data_y = pd.read_csv(
        calibration_directory + calibration_session + x_axis)

    light_posit_ref = np.array(
        [stats.mean(data_ref.iloc[:, 2]), stats.mean(data_ref.iloc[:, 3]), stats.mean(data_ref.iloc[:, 4]), 1])
    light_posit_x = np.array(
        [stats.mean(data_x.iloc[:, 2]), stats.mean(data_x.iloc[:, 3]), stats.mean(data_x.iloc[:, 4]), 1])
    light_posit_y = np.array(
        [stats.mean(data_y.iloc[:, 2]), stats.mean(data_y.iloc[:, 3]), stats.mean(data_y.iloc[:, 4]), 1])

    light_posit_refx = light_posit_x[0:3] - light_posit_ref[0:3]
    light_posit_refy = light_posit_y[0:3] - light_posit_ref[0:3]

    light_posit_refortho = np.cross(light_posit_refx, light_posit_refy)
    light_posit_ortho = light_posit_ref[0:3] + light_posit_refortho

    light_posit_refx, light_posit_refy, light_posit_refortho, light_posit_ortho = list(
        map(np.append, [light_posit_refx, light_posit_refy, light_posit_refortho, light_posit_ortho], [1] * 4))

    # The orthogonal vectors point upwards in the arena, I think it makes sense, check with Antonin

    aruc_posit_ref = np.array([0, 0, 0, 1])
    aruc_posit_x = np.array([30, 0, 0, 1])
    aruc_posit_y = np.array([0, 30, 0, 1])
    #Because the magnitude of a cross product is the area of the paralellogram that has the terms of the product as
    #sides, and posit_x and posit_y are perpendicular, then the magnitude of posit_ortho is 30^2=900
    aruc_posit_ortho = np.array([0, 0, 900, 1])

    aruc_matrix = np.transpose([aruc_posit_ref, aruc_posit_x, aruc_posit_y, aruc_posit_ortho])
    light_matrix = np.transpose([light_posit_ref, light_posit_x, light_posit_y, light_posit_ortho])

    # The result is obtained by multiplying the desired set by the inverted original set

    transform_matrix = np.matmul(aruc_matrix, np.linalg.inv(light_matrix))
    return transform_matrix

def transform_data(data, transform_matrix):
    trans_data = data

    for index, row in data.iterrows():
        point_vector = np.array([row[2], row[3], row[4]])
        point_vector = np.append(point_vector, 1)
        trans_point = np.matmul(transform_matrix, np.transpose(point_vector))
        trans_data.iloc[index, 2], trans_data.iloc[index, 3], trans_data.iloc[index, 4] = trans_point[0], trans_point[1],trans_point[2]

    return trans_data

def plot_single_occupancy(data):

    plt.close('all')

    plt.figure(figsize=(6.8, 4.2))
    plt.scatter(data.iloc[:, 2], data.iloc[:, 3], c=data.iloc[:, 4], cmap="magma", s=0.5)
    plt.title("Lighthouse projection of occupancy in the x,y plane")
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.colorbar()

    plt.figure(figsize=(6.8, 4.2))
    plt.scatter(data.iloc[:, 3], data.iloc[:, 4], c=data.iloc[:, 2], cmap="magma", s=0.5)
    plt.title("Lighthouse projection of occupancy in the y,z plane")
    plt.xlabel("Y axis")
    plt.ylabel("Z axis")
    plt.gca().invert_yaxis()
    plt.colorbar()

    plt.figure(figsize=(6.8, 4.2))
    plt.scatter(data.iloc[:, 2], data.iloc[:, 4], c=data.iloc[:, 3], cmap="magma", s=0.5)
    plt.title("Lighthouse projection of occupancy in the x,z plane")
    plt.xlabel("X axis")
    plt.ylabel("Z axis")
    plt.gca().invert_yaxis()
    plt.colorbar()