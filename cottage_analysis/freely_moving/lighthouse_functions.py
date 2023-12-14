import numpy as np
import statistics as stats


def obtain_transform_matrix(data_ref, data_x, data_y, calib_length):
    """Calculates the transformation matrix between the lighthouse and the aruco system

    Args:
        data_ref (pandas dataframe): dataframe containing the position of the lighthouse
            in the aruco system when the lighthouse is placed in the origin
        data_x (pandas dataframe): dataframe containing the position of the lighthouse
            in the aruco system when the lighthouse is placed in the x axis
        data_y (pandas dataframe): dataframe containing the position of the lighthouse
            in the aruco system when the lighthouse is placed in the y axis
        calib_length (float): length of the axis in which the lighthouse is placed
    """
    light_posit_ref = data_ref[['x', 'y', 'z']].mean(axis=0).values
    light_posit_x = data_x[['x', 'y', 'z']].mean(axis=0).values
    light_posit_y = data_y[['x', 'y', 'z']].mean(axis=0).values

    light_posit_refx = light_posit_x - light_posit_ref
    light_posit_refy = light_posit_y - light_posit_ref

    light_posit_refortho = np.cross(light_posit_refx, light_posit_refy)
    norm_fact = (calib_length * np.linalg.norm(light_posit_refx)) / np.linalg.norm(
        light_posit_refortho
    )

    light_posit_refortho_scaled = light_posit_refortho * norm_fact
    light_posit_ortho = light_posit_ref + light_posit_refortho_scaled

    light_posit_x, light_posit_y, light_posit_ref, light_posit_ortho = list(
        map(
            np.append,
            [light_posit_x, light_posit_y, light_posit_ref, light_posit_ortho],
            [1] * 4,
        )
    )

    # The orthogonal vectors point upwards in the arena, I think it makes sense, check with Antonin
    aruc_posit_ref = np.array([0, 0, 0, 1])
    aruc_posit_x = np.array([calib_length, 0, 0, 1])
    aruc_posit_y = np.array([0, calib_length, 0, 1])
    aruc_posit_ortho = np.cross(aruc_posit_x[0:3], aruc_posit_y[0:3])
    aruc_posit_ortho = np.append(aruc_posit_ortho, 1)

    aruc_matrix = np.transpose(
        [aruc_posit_ref, aruc_posit_x, aruc_posit_y, aruc_posit_ortho]
    )
    light_matrix = np.transpose(
        [light_posit_ref, light_posit_x, light_posit_y, light_posit_ortho]
    )

    # The result is obtained by multiplying the desired set by the inverted original set
    transform_matrix = np.matmul(aruc_matrix, np.linalg.inv(light_matrix))
    return transform_matrix



def transform_data(data, transform_matrix):
    """Transforms the data from the lighthouse system to the aruco system

    Args:
        data (pandas dataframe): dataframe containing the data to be transformed
        transform_matrix (numpy array): transformation matrix between the lighthouse
            and the aruco system

    Returns:
        pandas dataframe: dataframe containing the transformed data
    """
    trans_data = data.copy()
    for index, row in trans_data.iterrows():
        point_vector = row[['x', 'y', 'z']].values
        point_vector = np.append(point_vector, 1)
        trans_point = np.matmul(transform_matrix, np.transpose(point_vector))
        (
            trans_data.iloc[index, 2],
            trans_data.iloc[index, 3],
            trans_data.iloc[index, 4],
        ) = (trans_point[0], trans_point[1], trans_point[2])

    return trans_data
