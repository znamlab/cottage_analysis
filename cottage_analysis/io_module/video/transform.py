#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 14:08:43 2021

@author: hey2

Transform the shape of data
"""
import numpy as np


def find_idx_colMajor(idx, R, C, F):
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



def colMajor_to_frameMajor(idx, R, C, F):
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
    [r,c,f] = find_idx_colMajor(idx, R, C, F)
    
    new_idx = (r*C+c)*F + f
    
    return new_idx

