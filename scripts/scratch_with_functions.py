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

#Now we transform every point of the recording and repeat the plots.

calibration_directory = "/Users/antoniocolas/code/cottage_rawdata/lighthouse_calibration/"
calibration_session = "S20220808"

reference, y_axis, x_axis = ["/calibration_reference/ts4231-1_2022-08-08T16_31_57.csv", "/calibration_xaxis/ts4231-1_2022-08-08T16_33_05.csv", "/calibration_yaxis/ts4231-1_2022-08-08T16_33_54.csv"]

#CAREFUL I've switched the x and y datasets on purpose because they were really inverted: one was the other. Talk to Antonin

tran_matrix = lif.obtain_transform_matrix(calibration_directory, calibration_session, x_axis, y_axis, reference, 30)

#t_data_1, t_data_2 = transform_data(data_1, tran_matrix), transform_data(data_2, tran_matrix)

t_data_1 = lif.transform_data(data_1, tran_matrix)

lif.plot_single_occupancy(t_data_1, colormap=False)

print('hello world')
