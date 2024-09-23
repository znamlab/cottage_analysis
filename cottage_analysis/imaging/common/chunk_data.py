#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 15 21:30:19 2021

@author: hey2

Chunk data 

"""


def chunk_data(arr, chunk_size, chunk_nodes=None):
    """
    Chunk an array with a fixed size or with determined nodes.

    Parameters
    ----------
    arr : array or list
        The original array or list to be chunked.
    chunk_size : int or None
        The size of each chunk or None for non-fixed size.
    chunk_nodes : list or None, optional
        The list of nodes for the chunks or None for fixed size.

    Returns
    -------
    chunk_list : list
        2D list of the chunked array.

    """

    # chunk data with determined nodes (no fixed size), the nodes should include the last element
    if chunk_size == None:
        chunk_nodes_idx = [
            idx for idx, element in enumerate(arr) if (element in chunk_nodes)
        ]
        chunk_list = []
        for i in range(0, len(chunk_nodes_idx) - 1):
            chunk_list.append(arr[chunk_nodes_idx[i] : chunk_nodes_idx[i + 1]])

    # chunk data with fixed size
    else:
        chunk_groups = int(len(arr) / chunk_size)
        chunk_list = []  # list of chunks with their start & stop index
        for i in range(0, chunk_groups):
            chunk_list.append(arr[i * chunk_size : (i + 1) * chunk_size])
        chunk_list.append(arr[chunk_groups * chunk_size :])

    return chunk_list
