print('Small notes on indexing')
msg = """
Bonsai saves data in Column major order (a.k.a. Fortran order). So a 4 columns x 3 rows video with 2 frame like that:
frame1 = [[a1.1, a1.2, a1.3, a1.4],  and frame2 = [[b1.1, b1.2, b1.3, b1.4],
      [a2.1, a2.2, a2.3, a2.4],                [b2.1, b2.2, b2.3, b2.4],
      [a3.1, a3.2, a3.3, a3.4]]                [b3.1, b3.2, b3.3, b3.4]]

would be saved on disk as:
data = [a1.1, a2.1, a3.1, a1.2, a2.2, a3.2, [...], a3.4, b1.1, b2.1, b3.1, [...], b2.4, b3.4]

We want to read this data (either in C or F order) and reshape it so that the pixel information across frames is 
contiguous. 
How do we do that?

"""
print(msg)
import numpy as np
Ncolumns = 4
Nrows = 3
Nframes = 2
data = np.arange(Nrows * Ncolumns * Nframes)
print('Create some contiguous data')
print(data)

print('Make a data frame as saved by bonsai: Fortran = column major')
f_data = data.reshape([Ncolumns, Nrows, Nframes], order='F')
print('First frame (data[:, :, 0):')
print(f_data[:, :, 0])
print('\nThis just means that:\nd[d1,d2,d3] = data[d1 + N1 * d2 + N1 * N2 * d3], '
      'where Nx is the shape of dimension x')

def func_column_major(d1, d2, d3, N1, N2):
    """First dimension is contiguous"""
    return d1 + N1 * d2 + N1 * N2 * d3

index = func_column_major(0, 1, 0, N1=Ncolumns, N2=Nrows)
print('Do it explicitly for d[0,1,0]: index = %d' % index)
index = func_column_major(3, 2, 0, N1=Ncolumns, N2=Nrows)
print('And for d[3,2,0]: index = %d' % index)
print()
print('Note that we can also read the same data as Row major = C order')
print('We just need to change the shape to Nframes x Nrows x Ncolumns')

c_data = data.reshape([Nframes, Nrows, Ncolumns], order='C')
print('First frame (data[0, :, :]):')
print(c_data[0, :, :])
print('This just means that:')
print('d[d1,d2,d3] = data[d3 + N3 * d2 + N3 * N2 * d1], where Nx is the shape of dimension x')

def func_row_major(d1, d2, d3, N2, N3):
    """Last dimension is contiguous"""
    return d3 + N3 * d2 + N3 * N2 * d1
index = func_row_major(0, 1, 0, N2=Nrows, N3=Ncolumns)
print('Do it explicitly for d[0,1,0]: index = %d' % index)
index = func_row_major(0, 2, 2, N2=Nrows, N3=Ncolumns)
print('And for d[0,2,2]: index = %d' % index)
print()
print('We want a C-order (numpy default) array with the pixel information being contiguous on disk')
print('That means that the shape of the array should end with the frame dimension.')
print('We can check that by looking as c_data.swapaxes(0, 2).flatten():')
print(c_data.swapaxes(0, 2).flatten())


#%% Test to transform data into frame-major
ammp_filename = '/Users/hey2/Desktop/f_data.npy'
newmmp_filename = '/Users/hey2/Desktop/new_data.npy'
Nrows = 3  # Height
Ncolumns = 4  # Width
Nframes = 2
f_data = np.arange(Nrows * Ncolumns * Nframes)  # data written by Bonsai in disk
video = np.swapaxes(f_data.reshape([Nframes,Ncolumns,Nrows]),0,2)  # read video data



# bonsai saves video in C-order 
def Fidx_to_Ccoord(idx, R, C, F):
    '''
    Convert the index of data saved in F-order (Bonsai) to C-order coordinate

    Parameters
    ----------
    idx : int
        F-order index of data
    R : int
        Total number of rows
    C : int
        Total number of columns
    F : int
        Total number of frames

    Returns
    -------
    [r.c.f]: [row_idx, col_idx, frame_idx] in C-order

    '''
    
    f = int(idx/(R*C))
    c = int((idx-f*(R*C))/R)
    r = (idx-R*C*f) - c*R
    
    return[r,c,f]



# C-order to frame-major index
def Ccoord_to_FrameMajorIdx(Ccoord, R, C, F):
    '''
    Convert C-order coordinate to the index of frame-major data to be saved

    Parameters
    ----------
    Ccoord : [r,c,f]
        [row_idx, col_idx, frame_idx] in C-order
    R : int
        Total number of rows
    C : int
        Total number of columns
    F : int
        Total number of frames

    Returns
    -------
    idx: int
        Index of frame-major data to be saved 

    '''
    
    [r,c,f] = Ccoord
    idx = (r*C+c)*F + f
    
    return idx
    


def Fidx_to_FrameMajorIdx(idx, R, C, F):
    '''
    Convert the index of data saved in F-order (Bonsai) to the index of frame-major data to be saved

    Parameters
    ----------
    idx : int
        F-order index of data
    R : int
        Total number of rows
    C : int
        Total number of columns
    F : int
        Total number of frames

    Returns
    -------
    new_idx : int
        Index of frame-major data to be saved 

    '''
    f = int(idx/(R*C))
    c = int((idx-f*(R*C))/R)
    r = (idx-R*C*f) - c*R
    
    new_idx = (r*C+c)*F + f
    
    return new_idx



new_data = np.zeros(len(f_data))
for i in range(0,len(f_data)):
    new_data[Fidx_to_FrameMajorIdx(i, R=Nrows, C=Ncolumns, F=Nframes)] = f_data[i]  


fp = np.memmap(ammp_filename, dtype='float32', mode='w+', shape=f_data.shape)
fp[:] = f_data[:]
fp.flush()

new_fp = np.memmap(newmmp_filename, dtype='float32', mode='w+', shape=new_data.shape)
new_fp[:] = new_data[:]
new_fp.flush()

check_fp = np.memmap(ammp_filename, dtype='float32', mode='r', shape=f_data.shape)
check_new_fp = np.memmap(newmmp_filename, dtype='float32', mode='r', shape=new_data.shape)
